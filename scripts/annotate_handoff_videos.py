#!/usr/bin/env python3
"""Label who controls the arm in every Sol + G0.5 hand-off rollout video.

The video has one frame per control step (pyrua_eval's SimArm). Steps run in order: the
initial hold (45 steps), then every stateful PyRUA call's ``env_steps`` from
pyrua/calls.jsonl, then the hold after the program finished. That accounting reproduces the
logged "program finished at native step N" exactly, so each vla_grasp / vla_place maps to its
frames. Writes <dir>/annotated/epNNN.mp4 (banner: SOL / G0.5 GRASP / G0.5 PLACE / HOLD) and
<dir>/annotated/index.md (times of every segment, result).

    python3 scripts/annotate_handoff_videos.py outputs/llm_harness_h2h/pyrua_sol_g05_handoff_20261015_162536
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from llm_eval_results import latest_results
from video_provenance import fingerprint, current, write_manifest, harness

FPS = 30
HOLD_STEPS = 45
COLORS = {"SOL": "0x3b82f6", "SOL APPROACH": "0x3b82f6", "SOL PLACING": "0x3b82f6", "SOL TAKEOVER": "0xdc2626",
          "TAKEOVER -> NEUTRAL": "0xdc2626", "BACK TO G0.5 -> NEUTRAL": "0x7c3aed", "END -> HOME": "0x6b7280",
          "SOL TAKEOVER (SLOW)": "0xdc2626", "G0.5": "0xf97316", "G0.5 GRASP": "0xf59e0b", "G0.5 PLACE": "0x10b981", "G0.5 DOCK": "0xa855f7", "G0.5 PICK+PLACE": "0xec4899", "G0.5 BIN PLACE": "0x14b8a6", "G0.5 BIN CYCLE": "0xf97316", "HOLD": "0x6b7280"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def esc(text: str) -> str:
    """Text for an ffmpeg drawtext inside a filter graph."""
    return (text.replace("\\", "").replace("'", "").replace(":", "\\:").replace(",", "\\,")
            )


VERDICT_COLORS = {"progress": "0x4b5563", "failed_attempt": "0xd97706", "stuck": "0xdc2626", "placed": "0x16a34a",
                  "lifted": "0x16a34a", None: "0x2563eb"}
FLASH_FRAMES = 45  # Sol's verdict shows for 1.5 s from the pause it was made at


def verdict_text(k: dict) -> str:
    v, obj = k.get("verdict"), k.get("object")
    if k.get("decision") == "first_handoff":
        text = "Sol hands the arm to G0.5"
    elif k.get("decision") == "restart_policy":
        text = f"Sol: PLACED it itself ({k.get('in_bin')} in bin) -> back to G0.5"
    elif v == "failed_attempt":
        text = f"Sol: FAILED ATTEMPT #{k.get('attempts')} on {obj}"
    elif v == "stuck":
        text = f"Sol: STUCK on {obj} ({k.get('stuck_s')} s)"
    elif v == "placed":
        text = f"Sol: PLACED in bin ({k.get('in_bin')} so far)"
    elif v == "lifted" and k.get("decision") == "continue":
        text = "Sol: LIFTED (G0.5 carrying it: no takeover)"  # v9.3 bin tasks
    elif v == "lifted":
        text = "Sol: LIFTED -> Sol takes over the placement"
    else:
        text = f"Sol: {v}"
    if k.get("decision") == "takeover":
        text += f"  -> TAKEOVER ({k.get('reason')})"
    return text


def segments(ep: Path, n_frames: int):
    """Who drives each frame, plus Sol's verdict flashes and the running HUD.

    Returns (segments, flashes, huds): segments (label, first, last, note) cover the video;
    flashes (text, color, first, last) show each verdict from the pause it was made at; huds
    (text, first, last) show Sol's running tally (objects it reported in the bin, failed attempts).
    """

    out = [("HOLD", 0, HOLD_STEPS - 1, "initial hold")]
    flashes, tally = [], []
    step = HOLD_STEPS
    calls = [json.loads(line) for line in (ep / "pyrua" / "calls.jsonl").read_text().splitlines() if line.strip()]
    v9 = any(c["tool"] == "vla_run" for c in calls)
    spatial_first = v9 and any(c["tool"] == "sol_verdict" and c["kwargs"].get("decision") == "first_handoff" for c in calls)
    sol_label = "SOL APPROACH" if spatial_first else "SOL"
    routine, routine_left = None, 0
    fails: dict[str, int] = {}
    in_bin = 0
    for call in calls:
        tool, kw = call["tool"], call.get("kwargs") or {}
        if tool == "sol_verdict":
            text = verdict_text(kw)
            color = "0xb91c1c" if kw.get("decision") == "takeover" else VERDICT_COLORS.get(kw.get("verdict"), "0x2563eb")
            flashes.append([text, color, step, step + FLASH_FRAMES - 1])
            if kw.get("verdict") == "failed_attempt" and kw.get("object"):
                fails[kw["object"]] = int(kw.get("attempts") or 0)
            in_bin = int(kw.get("in_bin") or in_bin)
            if kw.get("decision") == "takeover":
                # v9 opens and goes to the start pose (the next grip + home); v9.1 takes over in place
                in_place = any(c["tool"] == "takeover_slow" for c in calls)
                routine, routine_left, sol_label = (None, 0, "SOL TAKEOVER (SLOW)") if in_place else (
                    "TAKEOVER -> NEUTRAL", 3, "SOL TAKEOVER")
            elif kw.get("decision") == "restart_policy":
                routine, routine_left, sol_label = "BACK TO G0.5 -> NEUTRAL", 3, "SOL"
            elif kw.get("decision") == "lifted" or kw.get("verdict") == "lifted":
                sol_label = "SOL PLACING"
            parts = [] if spatial_first else [f"Sol-reported in bin {in_bin}"]  # non-bin: no bin count
            if fails:
                parts.append("failed: " + ", ".join(f"{o} x{n}" for o, n in fails.items()))
            if parts:
                tally.append((step, "  |  ".join(parts)))
            continue
        if tool == "end_episode":
            routine, routine_left = "END -> HOME", 3
            continue
        n = int(call.get("env_steps") or 0) if call.get("stateful") else 0
        if not n:
            continue
        if tool == "vla_run":
            label, routine_left = "G0.5", 0
        elif routine_left and tool in ("grip", "home"):
            label, routine_left = routine, routine_left - 1
        else:
            routine_left = 0
            label = {"vla_grasp": "G0.5 GRASP", "vla_place": "G0.5 PLACE", "vla_dock": "G0.5 DOCK",
                     "vla_pick_place": "G0.5 PICK+PLACE", "vla_bin_place": "G0.5 BIN PLACE",
                     "vla_bin_cycle": "G0.5 BIN CYCLE"}.get(tool, sol_label)
        summ = call.get("summary") or {}
        note = tool if not label.startswith("G0.5") else f"{tool} holding={summ.get('holding')}"
        if out and out[-1][0] == label and not label.startswith("HOLD"):  # merge consecutive stretches
            out[-1] = (label, out[-1][1], step + n - 1, out[-1][3])
        else:
            out.append((label, step, step + n - 1, note))
        step += n
    if step < n_frames:
        out.append(("HOLD", step, n_frames - 1, "program finished; benchmark judging"))
    for i in range(len(flashes) - 1):  # a newer verdict replaces the one on screen
        flashes[i][3] = min(flashes[i][3], flashes[i + 1][2] - 1)
    huds = [(text, a, (tally[i + 1][0] - 1) if i + 1 < len(tally) else max(n_frames, step) + 10_000)
            for i, (a, text) in enumerate(tally)]
    return out, [tuple(f) for f in flashes if f[3] >= f[2]], huds


CAPTION_FRAMES = 90  # a Sol message stays on screen 3 s (or until its next message)
CAPTION_CHARS = 105  # per line at fontsize 18 across 1280 px; two lines at most


def sol_messages(ep: Path) -> list[tuple[int, str]]:
    """Sol's own messages (its free-text judgements), each at the video frame it was made at.

    Codex events list Sol's messages and code cells in order; pyrua/tool_turns.json gives each
    cell's native steps (the video has one frame per step, after the 45-step initial hold), so a
    message made between two cells sits at the frame where the earlier cell's motion ended.
    """

    events, turns = ep / "pyrua" / "codex_events.jsonl", ep / "pyrua" / "tool_turns.json"
    if not events.exists() or not turns.exists():
        return []
    steps = [int(t.get("env_steps") or 0) for t in json.loads(turns.read_text())]
    out, cursor, k = [], HOLD_STEPS, 0
    for line in events.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = e.get("item") or {}
        if e.get("type") != "item.completed":
            continue
        if item.get("type") == "agent_message" and item.get("text", "").strip():
            text = " ".join(item["text"].split())
            if out and out[-1][0] == cursor:  # several messages with no motion between: one caption
                out[-1] = (cursor, out[-1][1] + " " + text)
            else:
                out.append((cursor, text))
        elif item.get("type") == "mcp_tool_call":
            cursor += steps[k] if k < len(steps) else 0
            k += 1
    return out


def wrap(text: str) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > CAPTION_CHARS:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    if len(lines) > 2:
        lines = [lines[0], lines[1][: CAPTION_CHARS - 3] + "..."]
    return lines


def tc(frame: int) -> str:
    s = frame / FPS
    return f"{int(s // 60)}:{s % 60:04.1f}"


def annotate(ep: Path, out_dir: Path, hybrid: bool = True) -> str | None:
    video = ep / "rollout_overhead_wrist.mp4"
    result = ep / "result.json"
    if not video.exists() or not result.exists() or not (ep / "pyrua" / "calls.jsonl").exists():
        return None
    target = out_dir / f"{ep.name}.mp4"
    r = json.loads(result.read_text())
    n_frames = int(r.get("agent_control_steps") or 0)
    segs, flashes, huds = segments(ep, n_frames)
    verdict = "SUCCESS" if r.get("success") else f"FAIL ({r.get('termination_reason')})"
    extra = f"  in bin {r.get('objects_in_bin')}/{r.get('bin_object_count')}" if r.get("objects_in_bin") is not None else ""
    source_files = [video, result, ep/'pyrua/calls.jsonl', Path(__file__),
                    Path(__file__).with_name('video_provenance.py')]
    source_files += [p for p in (ep/'pyrua/codex_events.jsonl', ep/'pyrua/tool_turns.json') if p.exists()]
    source_files += [p for p in (ep/'pyrua/prompt.txt',) if p.exists()]
    provenance = {'hybrid': hybrid, 'harness': harness(r, hybrid),
                  'sources': [fingerprint(p) for p in source_files]}
    manifest = target.with_suffix('.annotation.json')
    if not current(target, manifest, provenance):
        filters = ["drawbox=x=0:y=0:w=iw:h=44:color=black@0.55:t=fill"]
        for label, a, b, _ in segs:
            filters.append(
                f"drawtext=fontfile={FONT}:text='{esc(label)}':x=12:y=8:fontsize=26:fontcolor=white:expansion=none:"
                f"box=1:boxcolor={COLORS[label]}@0.9:boxborderw=6:enable='between(n\\,{a}\\,{b})'"
            )
        for text, color, a, b in flashes:  # Sol's verdict at the pause it was made at
            filters.append(
                f"drawtext=fontfile={FONT}:text='{esc(text)}':x=12:y=58:fontsize=24:fontcolor=white:expansion=none:"
                f"box=1:boxcolor={color}@0.92:boxborderw=8:enable='between(n\\,{a}\\,{b})'"
            )
        for text, a, b in huds:  # Sol's running tally
            filters.append(
                f"drawtext=fontfile={FONT}:text='{esc(text)}':x=12:y=h-34:fontsize=20:fontcolor=white:expansion=none:"
                f"box=1:boxcolor=black@0.6:boxborderw=6:enable='between(n\\,{a}\\,{b})'"
            )
        msgs = sol_messages(ep)
        for i, (a, text) in enumerate(msgs):  # Sol's own words, at the frame it said them
            b = a + CAPTION_FRAMES - 1
            if i + 1 < len(msgs):
                b = min(b, msgs[i + 1][0] - 1)
            lines = wrap("Sol: " + text)
            for j, ln in enumerate(lines):
                y = f"h-{86 - 26 * j + (26 if len(lines) == 1 else 0)}"
                filters.append(
                    f"drawtext=fontfile={FONT}:text='{esc(ln)}':x=12:y={y}:fontsize=18:fontcolor=white:expansion=none:"
                    f"box=1:boxcolor=0x1e3a8a@0.85:boxborderw=5:enable='between(n\\,{a}\\,{b})'"
                )
        header = esc(f"{ep.name} | PyRUA {harness(r, hybrid)} | FINAL {verdict}{extra}")
        filters.append(f"drawtext=fontfile={FONT}:text='{header}':x=w-tw-12:y=12:fontsize=18:fontcolor=white:expansion=none")
        script = out_dir / f".{ep.name}.filter"
        script.write_text(",".join(filters))
        temporary = target.with_name(target.stem+'.building.mp4')
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "2", "-i", str(video), "-filter_script:v", str(script),
                        "-c:v", "libx264", "-threads", "2", "-crf", "23", "-preset", "veryfast", str(temporary)], check=True)
        temporary.replace(target)
        write_manifest(manifest, provenance)
    rows = "; ".join(f"{label} {tc(a)}–{tc(b + 1)}" for label, a, b, _ in segs if label != "HOLD" or a > 0)
    g = sum(1 for s in segs if s[0].startswith("G0.5"))
    takeovers = sum(1 for f in flashes if "TAKEOVER" in f[0])
    fails = sum(1 for f in flashes if "FAILED ATTEMPT" in f[0])
    return f"| {ep.name} | {r.get('instruction', '')[:60]} | {verdict}{extra} | {g} G0.5 stretches, {fails} failed attempts (Sol), {takeovers} takeovers | {rows} |"


def main() -> None:
    base = Path(sys.argv[1])
    out_dir = base / "annotated"
    out_dir.mkdir(exist_ok=True)
    lines = ["# Sol + G0.5 hand-off videos", "",
             "Banner: SOL (blue) = Sol's own motions (incl. the harness's takeover move to the start pose), G0.5 (orange) = G0.5 driving (Sol's looks pause physics, so they take no video time), G0.5 GRASP (amber), G0.5 PLACE (green), G0.5 DOCK (purple), G0.5 PICK+PLACE (pink), G0.5 BIN PLACE (teal), G0.5 BIN CYCLE (orange), HOLD (grey) = "
             "initial hold or program finished. Times are video time (30 fps).", "",
             "| Episode | Instruction | Result | Hand-offs (Sol's verdicts) | Segments |", "|---|---|---|---|---|"]
    for _, (result_path, _result) in sorted(latest_results([base]).items()):
        row = annotate(result_path.parent, out_dir)
        if row:
            lines.append(row)
    (out_dir / "index.md").write_text("\n".join(lines) + "\n")
    print(f"{len(lines) - 6} episodes → {out_dir}")


if __name__ == "__main__":
    main()
