"""A credential-safe, metered HTTP transport for the small Inspect pilot."""

from __future__ import annotations

import json
from pathlib import Path
import httpx


class BudgetExceeded(RuntimeError):
    """No further paid requests are allowed by the pilot's local budget."""


class BudgetTransport(httpx.BaseTransport):
    """Account actual Responses usage and reserve a conservative next-call cost.

    Limits are local estimates at pinned standard prices, not account billing
    controls. No credentials, request headers, or raw images enter this ledger.
    """

    def __init__(self, ledger: Path, total_limit=15.0, trial_limit=2.0, inner=None):
        self.ledger = ledger
        self.total_limit = total_limit
        self.trial_limit = trial_limit
        self.inner = inner or httpx.HTTPTransport(retries=0)
        self.trial = "preflight"
        self.output_cap = 4096
        self._previous_input = None
        self._previous_tokens = 0
        self._previous_trial = None
        self._previous_context = None

    def spent(self, trial=None):
        if not self.ledger.exists():
            return 0.0
        rows = [json.loads(s) for s in self.ledger.read_text().splitlines() if s.strip()]
        return sum(r["estimated_usd"] for r in rows if trial is None or r["trial"] == trial)

    def handle_request(self, request):
        body = json.loads(request.content)
        assert request.url.host == "api.openai.com", "Pilot credential is restricted to OpenAI"
        assert body["model"] == "gpt-6.1-sol"
        body["max_output_tokens"] = self.output_cap
        # Assume all input misses cache. 6144 tokens/image and one token per
        # UTF-8 text byte are conservative reserves for our fixed 640x480 views.
        images = 0

        def text_bytes(value):
            nonlocal images
            if isinstance(value, dict):
                if value.get("type") == "input_image":
                    images += 1
                    return 0
                return sum(text_bytes(v) for v in value.values())
            if isinstance(value, list):
                return sum(text_bytes(v) for v in value)
            return len(value.encode()) if isinstance(value, str) else 0

        # Recounting every historical image at its worst-case bound eventually
        # stops cheap cached conversations far below the dollar limit. Once the
        # provider has measured an input, retain its FULL token count (not just
        # uncached tokens) and bound only the changed/appended message suffix.
        # The old changed suffix is deliberately not subtracted, so this still
        # over-reserves if a cache marker or message serialization changes.
        context = {k: v for k, v in body.items() if k not in ('input', 'max_output_tokens')}
        incoming = body.get('input')
        if (self._previous_trial == self.trial and self._previous_context == context
                and isinstance(incoming, list) and self._previous_input is not None):
            shared = 0
            for old, new in zip(self._previous_input, incoming):
                if old != new:
                    break
                shared += 1
            if shared:
                reserved_tokens = self._previous_tokens + text_bytes(incoming[shared:]) + 6144 * images + 1024
            else:
                reserved_tokens = text_bytes(body) + 6144 * images
        else:
            reserved_tokens = text_bytes(body) + 6144 * images
        long_input = reserved_tokens > 272000
        reserve = (reserved_tokens * (4 if long_input else 2) +
                   self.output_cap * (15 if long_input else 10)) / 1e6
        if (self.spent() + reserve > self.total_limit or
                self.spent(self.trial) + reserve > self.trial_limit):
            raise BudgetExceeded("Pilot budget reserve reached; no request was sent")
        effective = httpx.Request(request.method, request.url, headers=request.headers,
                                  content=json.dumps(body).encode(), extensions=request.extensions)
        # A replaced body needs a fresh length rather than the caller's length.
        effective.headers["Content-Length"] = str(len(effective.content))
        response = self.inner.handle_request(effective)
        response.read()
        if response.status_code >= 400:
            # Record machine-readable provider errors without messages, headers,
            # credentials or request images. The campaign can stop on hard quota
            # while allowing ordinary per-minute rate limits to retry.
            try:
                failure = response.json().get('error') or {}
                failure = failure if isinstance(failure, dict) else {}
            except ValueError:
                failure = {}
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.with_suffix('.errors.jsonl').open('a') as f:
                f.write(json.dumps({'trial': self.trial, 'http_status': response.status_code,
                                   'error_code': failure.get('code'), 'error_type': failure.get('type')})+'\n')
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage") or {}
            inp = int(usage.get("input_tokens", 0))
            cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
            writes = int((usage.get("input_tokens_details") or {}).get("cache_write_tokens", 0))
            out = int(usage.get("output_tokens", 0))
            input_factor, output_factor = (2, 1.5) if inp > 272000 else (1, 1)
            cost = ((max(0, inp-cached-writes)*2 + cached*0.1 + writes*2.5)*input_factor +
                    out*10*output_factor) / 1e6
            if not usage:
                cost = reserve
            row = {"trial": self.trial, "response_id": data.get("id"), "usage": usage,
                   "estimated_usd": cost, "service_tier": data.get("service_tier"),
                   "max_output_tokens": self.output_cap,
                   "reserved_input_tokens": reserved_tokens, "reserved_usd": reserve}
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a") as f:
                f.write(json.dumps(row) + "\n")
            if not usage:
                raise BudgetExceeded("Provider omitted usage; reserved cost recorded, stopping pilot")
            self._previous_input = incoming if isinstance(incoming, list) else None
            self._previous_tokens = inp
            self._previous_context = context
            self._previous_trial = self.trial
        return response

    def close(self):
        self.inner.close()
