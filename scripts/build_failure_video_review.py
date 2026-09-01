#!/usr/bin/env python3
"""Build an HTML review page with overhead clips for failed replay outcomes."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_KEY = "observation.images.overhead"


@dataclass(frozen=True)
class VideoSpan:
    path: Path
    start: float
    duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outcomes_jsonl", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--overwrite-clips",
        action="store_true",
        help="Re-encode clips that already exist.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A live collector may be in the middle of appending its last line.
                print(f"[WARN] Ignoring incomplete JSON at {path}:{line_number}")
    return records


def load_video_spans(dataset_root: Path) -> dict[int, VideoSpan]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read LeRobot episode metadata") from exc

    prefix = f"videos/{VIDEO_KEY}"
    columns = [
        "episode_index",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    ]
    spans: dict[int, VideoSpan] = {}
    parquet_paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No episode metadata under {dataset_root / 'meta' / 'episodes'}")

    for parquet_path in parquet_paths:
        data = pq.read_table(parquet_path, columns=columns).to_pydict()
        for row_index, episode_index in enumerate(data["episode_index"]):
            chunk_index = int(data[f"{prefix}/chunk_index"][row_index])
            file_index = int(data[f"{prefix}/file_index"][row_index])
            start = float(data[f"{prefix}/from_timestamp"][row_index])
            end = float(data[f"{prefix}/to_timestamp"][row_index])
            video_path = (
                dataset_root
                / "videos"
                / VIDEO_KEY
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            spans[int(episode_index)] = VideoSpan(video_path, start, end - start)
    return spans


def postmortem(record: dict[str, Any]) -> dict[str, Any]:
    attribution = (
        record.get("final_failure_attribution")
        or record.get("failure_attribution")
        or {}
    )
    return attribution.get("postmortem") or attribution.get("postmortem_raw") or {}


def clip_video(source: VideoSpan, destination: Path, overwrite: bool) -> str:
    if destination.is_file() and not overwrite:
        return f"[SKIP] {destination.name}"
    if not source.path.is_file():
        raise FileNotFoundError(source.path)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{source.start:.9f}",
        "-i",
        str(source.path),
        "-t",
        f"{source.duration:.9f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)
    return f"[OK] {destination.name}"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def evidence_rows(record: dict[str, Any], pm: dict[str, Any]) -> list[tuple[str, str]]:
    evidence = pm.get("evidence") or {}
    goal_metric = evidence.get("current_goal_metric") or {}
    rows = [
        ("Target", str(pm.get("target_object") or "unknown")),
        ("Strongest wrong object", str(pm.get("strongest_wrong_object") or "none")),
        ("Target manipulated", format_bool(pm.get("target_manipulated"))),
        ("Target acquired", format_bool(pm.get("target_acquired"))),
        ("Goal ever reached", format_bool(pm.get("goal_ever_reached"))),
        ("Goal ever confirmed", format_bool(pm.get("goal_ever_confirmed"))),
        ("Final goal met", format_bool(pm.get("final_goal_met"))),
        (
            "Maximum goal hold",
            f"{pm.get('max_goal_hold_steps', 'unknown')}/"
            f"{evidence.get('goal_required_confirmation_steps', 'unknown')} steps",
        ),
        ("Target maximum lift", f"{float(pm.get('target_lift_m') or 0.0):.4f} m"),
        (
            "Target maximum displacement",
            f"{float(pm.get('target_max_displacement_m') or 0.0):.4f} m",
        ),
        (
            "Final goal metric",
            json.dumps(goal_metric, sort_keys=True) if goal_metric else "unavailable",
        ),
    ]
    if evidence.get("ever_inside_bin_object_ids") is not None:
        rows.append(
            ("Objects ever inside bin", str(evidence.get("ever_inside_bin_object_ids")))
        )
    if evidence.get("never_manipulated_object_ids"):
        rows.append(
            ("Never manipulated object IDs", str(evidence["never_manipulated_object_ids"]))
        )
    return rows


def build_card(record: dict[str, Any]) -> str:
    benchmark = record.get("benchmark") or {}
    dataset = record.get("dataset") or {}
    label = record.get("label") or {}
    pm = postmortem(record)
    episode_index = int(dataset["episode_index"])
    scorer_reason = str(label.get("failure_reason") or "unknown")
    failure_type = str(pm.get("failure_type") or "unclassified")
    secondary = pm.get("secondary_failure_types") or []
    objects = benchmark.get("objects") or []
    rows = "".join(
        f"<dt>{esc(name)}</dt><dd>{esc(value)}</dd>"
        for name, value in evidence_rows(record, pm)
    )
    raw_payload = {
        "label": label,
        "final_failure_attribution": (
            record.get("final_failure_attribution")
            or record.get("failure_attribution")
        ),
        "best_achieved": record.get("best_achieved"),
        "closest_miss": record.get("closest_miss"),
    }
    raw_json = esc(json.dumps(raw_payload, indent=2, sort_keys=True))
    secondary_html = "".join(f"<span class='chip'>{esc(item)}</span>" for item in secondary)
    object_html = "".join(f"<span class='object'>{esc(item)}</span>" for item in objects)
    search_text = " ".join(
        [
            str(episode_index),
            str(benchmark.get("task_family") or ""),
            str(benchmark.get("instruction") or ""),
            " ".join(map(str, objects)),
            scorer_reason,
            failure_type,
            str(pm.get("rationale") or ""),
        ]
    ).lower()
    return f"""
      <article class="episode" data-family="{esc(benchmark.get('task_family') or 'unknown')}"
               data-reason="{esc(failure_type)}" data-search="{esc(search_text)}">
        <div class="episode-head">
          <div>
            <p class="eyebrow">Episode {episode_index:06d} · {esc(benchmark.get('task_family') or 'unknown')}</p>
            <h2>{esc(benchmark.get('instruction') or 'Unknown task')}</h2>
          </div>
          <span class="reason-badge">{esc(failure_type)}</span>
        </div>
        <div class="object-list">{object_html}</div>
        <div class="review-grid">
          <video controls preload="metadata" src="clips/episode_{episode_index:06d}_overhead.mp4"></video>
          <section class="diagnosis">
            <div class="scorer">
              <span>Exact scorer reason</span>
              <code>{esc(scorer_reason)}</code>
            </div>
            <p class="rationale">{esc(pm.get("rationale") or label.get("reason") or "No rationale recorded.")}</p>
            <div class="chips">{secondary_html}</div>
            <dl>{rows}</dl>
            <details>
              <summary>Full raw scoring evidence</summary>
              <pre>{raw_json}</pre>
            </details>
          </section>
        </div>
      </article>
    """


def build_html(
    failures: list[dict[str, Any]],
    completed_count: int,
    outcomes_path: Path,
) -> str:
    type_counts = Counter(postmortem(record).get("failure_type", "unclassified") for record in failures)
    family_counts = Counter(
        (record.get("benchmark") or {}).get("task_family", "unknown") for record in failures
    )
    type_options = "".join(
        f"<option value='{esc(name)}'>{esc(name)} ({count})</option>"
        for name, count in sorted(type_counts.items())
    )
    family_options = "".join(
        f"<option value='{esc(name)}'>{esc(name)} ({count})</option>"
        for name, count in sorted(family_counts.items())
    )
    cards = "\n".join(build_card(record) for record in failures)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 Failure Review</title>
  <style>
    :root {{
      --ink: #18201d;
      --muted: #64706a;
      --paper: #f5f0e7;
      --card: #fffdf8;
      --line: #d9d0c2;
      --red: #aa3428;
      --red-soft: #f6ddd5;
      --green: #295f4e;
      --shadow: 0 18px 50px rgba(47, 42, 32, .10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 8%, rgba(202, 131, 70, .16), transparent 28rem),
        linear-gradient(135deg, transparent 0 49%, rgba(41, 95, 78, .045) 49% 51%, transparent 51%) 0 0 / 28px 28px,
        var(--paper);
      font-family: "Avenir Next", "Gill Sans", sans-serif;
    }}
    header {{ max-width: 1440px; margin: auto; padding: 64px 28px 28px; }}
    h1 {{ margin: 0; max-width: 850px; font-family: Georgia, serif; font-size: clamp(2.7rem, 7vw, 6rem); line-height: .93; letter-spacing: -.055em; }}
    .lede {{ max-width: 790px; margin: 24px 0; color: var(--muted); font-size: 1.08rem; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .summary span {{ border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px; background: rgba(255,255,255,.55); }}
    .toolbar {{
      position: sticky; top: 0; z-index: 10; display: grid; grid-template-columns: 1fr 240px 190px;
      gap: 12px; padding: 14px max(28px, calc((100vw - 1384px) / 2));
      border-block: 1px solid var(--line); background: rgba(245, 240, 231, .92); backdrop-filter: blur(14px);
    }}
    input, select {{ width: 100%; border: 1px solid var(--line); border-radius: 8px; background: var(--card); padding: 11px 13px; color: var(--ink); font: inherit; }}
    main {{ max-width: 1440px; margin: auto; padding: 28px; }}
    .episode {{ margin-bottom: 28px; border: 1px solid var(--line); border-radius: 16px; background: var(--card); box-shadow: var(--shadow); overflow: hidden; }}
    .episode-head {{ display: flex; justify-content: space-between; gap: 20px; align-items: start; padding: 24px 24px 10px; }}
    .eyebrow {{ margin: 0 0 6px; color: var(--green); font-size: .78rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }}
    h2 {{ margin: 0; font-family: Georgia, serif; font-size: clamp(1.35rem, 3vw, 2.15rem); font-weight: 500; }}
    .reason-badge {{ flex: none; max-width: 300px; border-radius: 6px; padding: 8px 10px; color: var(--red); background: var(--red-soft); font-family: ui-monospace, monospace; font-size: .78rem; overflow-wrap: anywhere; }}
    .object-list {{ display: flex; flex-wrap: wrap; gap: 7px; padding: 4px 24px 22px; }}
    .object {{ border-left: 3px solid #c98246; padding: 3px 8px; background: #f3eadc; font-size: .86rem; }}
    .review-grid {{ display: grid; grid-template-columns: minmax(460px, 1.15fr) minmax(360px, .85fr); border-top: 1px solid var(--line); }}
    video {{ display: block; width: 100%; height: 100%; min-height: 360px; background: #111; object-fit: contain; }}
    .diagnosis {{ padding: 24px; border-left: 1px solid var(--line); }}
    .scorer span {{ display: block; margin-bottom: 7px; color: var(--muted); font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }}
    code {{ display: inline-block; color: var(--red); background: var(--red-soft); border-radius: 5px; padding: 5px 7px; overflow-wrap: anywhere; }}
    .rationale {{ margin: 18px 0; font-family: Georgia, serif; font-size: 1.18rem; line-height: 1.5; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 18px; }}
    .chip {{ border: 1px solid #dbb29e; border-radius: 999px; padding: 4px 8px; color: #7d3a2d; font-size: .75rem; }}
    dl {{ display: grid; grid-template-columns: 175px 1fr; margin: 0; font-size: .86rem; }}
    dt, dd {{ margin: 0; padding: 7px 0; border-top: 1px dotted var(--line); overflow-wrap: anywhere; }}
    dt {{ color: var(--muted); padding-right: 14px; }}
    details {{ margin-top: 18px; }}
    summary {{ cursor: pointer; color: var(--green); font-weight: 700; }}
    pre {{ max-height: 520px; overflow: auto; padding: 14px; border-radius: 8px; color: #e8eee9; background: #17221e; font-size: .72rem; white-space: pre-wrap; }}
    .empty {{ display: none; padding: 80px 20px; text-align: center; color: var(--muted); }}
    @media (max-width: 900px) {{
      .toolbar {{ grid-template-columns: 1fr; }}
      .review-grid {{ grid-template-columns: 1fr; }}
      .diagnosis {{ border-left: 0; border-top: 1px solid var(--line); }}
      video {{ min-height: 240px; }}
      .episode-head {{ display: block; }}
      .reason-badge {{ display: inline-block; margin-top: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <p class="eyebrow">SO-101 Bench · Overhead dataset camera</p>
    <h1>Failure review, frame by frame.</h1>
    <p class="lede">A snapshot of <strong>{len(failures)} failed episodes</strong> among {completed_count} completed outcomes. Every clip is cut from the original dataset video span and paired with the evaluator’s complete recorded diagnosis.</p>
    <div class="summary">
      <span>{len(failures)} failures</span>
      <span>{completed_count} completed</span>
      <span>{esc(outcomes_path.name)}</span>
    </div>
  </header>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search episode, object, task, or reason…">
    <select id="reason"><option value="">All failure types</option>{type_options}</select>
    <select id="family"><option value="">All task families</option>{family_options}</select>
  </div>
  <main>
    <div id="episodes">{cards}</div>
    <p class="empty" id="empty">No episodes match these filters.</p>
  </main>
  <script>
    const search = document.querySelector('#search');
    const reason = document.querySelector('#reason');
    const family = document.querySelector('#family');
    const cards = [...document.querySelectorAll('.episode')];
    const empty = document.querySelector('#empty');
    function filterCards() {{
      const needle = search.value.trim().toLowerCase();
      let shown = 0;
      for (const card of cards) {{
        const visible = (!needle || card.dataset.search.includes(needle))
          && (!reason.value || card.dataset.reason === reason.value)
          && (!family.value || card.dataset.family === family.value);
        card.hidden = !visible;
        if (visible) shown++;
        if (!visible) card.querySelector('video').pause();
      }}
      empty.style.display = shown ? 'none' : 'block';
    }}
    search.addEventListener('input', filterCards);
    reason.addEventListener('change', filterCards);
    family.addEventListener('change', filterCards);
    document.addEventListener('play', event => {{
      if (event.target.tagName !== 'VIDEO') return;
      for (const video of document.querySelectorAll('video')) {{
        if (video !== event.target) video.pause();
      }}
    }}, true);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.outcomes_jsonl)
    failures = [
        record for record in records if not (record.get("label") or {}).get("success", False)
    ]
    if not failures:
        raise RuntimeError(f"No failed outcomes found in {args.outcomes_jsonl}")

    # Freeze the live JSONL records before encoding so the page and clips agree.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    snapshot_path = args.output_dir / "failure_outcomes_snapshot.jsonl"
    snapshot_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in failures),
        encoding="utf-8",
    )

    spans = load_video_spans(args.dataset_root)
    jobs: list[tuple[VideoSpan, Path]] = []
    for record in failures:
        episode_index = int((record.get("dataset") or {})["episode_index"])
        try:
            span = spans[episode_index]
        except KeyError as exc:
            raise KeyError(f"No overhead video span for episode {episode_index}") from exc
        jobs.append((span, clips_dir / f"episode_{episode_index:06d}_overhead.mp4"))

    print(f"[INFO] Encoding {len(jobs)} overhead clips with {args.jobs} worker(s)")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(clip_video, span, destination, args.overwrite_clips)
            for span, destination in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)

    index_path = args.output_dir / "index.html"
    index_path.write_text(
        build_html(failures, len(records), args.outcomes_jsonl),
        encoding="utf-8",
    )
    print(f"[DONE] Review page: {index_path.resolve()}")


if __name__ == "__main__":
    main()
