from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).parents[1] / "so101_bench"
MODULE_NAMES = ("so101_bench", "so101_bench.utils", "so101_bench.utils.lerobot_calibration")
SAVED_MODULES = {name: sys.modules.get(name) for name in MODULE_NAMES}
try:
    package = ModuleType("so101_bench")
    package.__path__ = [str(PACKAGE_ROOT)]
    utils_package = ModuleType("so101_bench.utils")
    utils_package.__path__ = [str(PACKAGE_ROOT / "utils")]
    sys.modules["so101_bench"] = package
    sys.modules["so101_bench.utils"] = utils_package

    calibration_path = PACKAGE_ROOT / "utils" / "lerobot_calibration.py"
    calibration_spec = importlib.util.spec_from_file_location(
        "so101_bench.utils.lerobot_calibration",
        calibration_path,
    )
    assert calibration_spec is not None
    assert calibration_spec.loader is not None
    calibration_module = importlib.util.module_from_spec(calibration_spec)
    sys.modules[calibration_spec.name] = calibration_module
    calibration_spec.loader.exec_module(calibration_module)

    groot_path = PACKAGE_ROOT / "utils" / "groot.py"
    groot_spec = importlib.util.spec_from_file_location("so101_groot_profiler_test_module", groot_path)
    assert groot_spec is not None
    assert groot_spec.loader is not None
    groot_module = importlib.util.module_from_spec(groot_spec)
    sys.modules[groot_spec.name] = groot_module
    groot_spec.loader.exec_module(groot_module)
finally:
    for module_name, previous_module in SAVED_MODULES.items():
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

GR00TRemotePolicy = groot_module.GR00TRemotePolicy
MsgSerializer = groot_module.MsgSerializer
PolicyClient = groot_module.PolicyClient


class _FakeSocket:
    def __init__(self, response):
        self.response = response
        self.sent = None

    def send(self, message):
        self.sent = message

    def recv(self):
        return self.response


class _FakeActionClient:
    calls = 0

    def get_action(self, _observation):
        self.calls += 1
        action = {
            "single_arm": np.zeros((1, 2, 5), dtype=np.float32),
            "gripper": np.zeros((1, 2, 1), dtype=np.float32),
        }
        info = {"_timing": {"server": {"decode_s": 0.001, "handler_s": 0.25}}}
        return action, info


def test_policy_client_profiles_rpc_boundaries_and_bytes():
    timings = Counter()
    counters = Counter()
    client = PolicyClient.__new__(PolicyClient)
    client.api_token = None
    client.timing_callback = lambda name, value: timings.update({name: value})
    client.counter_callback = lambda name, value: counters.update({name: value})
    client.socket = _FakeSocket(MsgSerializer.to_bytes({"status": "ok"}))

    assert client.call_endpoint("ping", requires_input=False) == {"status": "ok"}
    assert timings["policy.rpc.ping.serialize"] >= 0.0
    assert timings["policy.rpc.ping.send"] >= 0.0
    assert timings["policy.rpc.ping.receive_wait"] >= 0.0
    assert timings["policy.rpc.ping.deserialize"] >= 0.0
    assert timings["policy.rpc.ping.total"] >= 0.0
    assert counters["policy.rpc.ping.request_bytes"] == len(client.socket.sent)
    assert counters["policy.rpc.ping.response_bytes"] == len(client.socket.response)


def test_remote_policy_profiles_query_and_cached_actions():
    timing_names = []
    counters = Counter()
    policy = GR00TRemotePolicy(
        device="cpu",
        cameras={},
        action_horizon=2,
        use_overhead_init=False,
        timing_callback=lambda name, _value: timing_names.append(name),
        counter_callback=lambda name, value: counters.update({name: value}),
    )
    client = _FakeActionClient()
    policy.client = client

    first = policy.get_action(torch.zeros(6), {})
    second = policy.get_action(torch.zeros(6), {})

    assert first.shape == (6,)
    assert second.shape == (6,)
    assert client.calls == 1
    assert counters["policy.action_cache.misses"] == 1
    assert counters["policy.action_cache.hits"] == 1
    assert counters["policy.actions_received"] == 2
    assert "policy.input.prepare_total" in timing_names
    assert "policy.server.handler_reported" in timing_names
    assert "policy.action.decode_chunk" in timing_names
    assert timing_names.count("policy.get_action.total") == 2
