#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sim_version="${SO101_SIM_VERSION:-4}"
if [[ "$sim_version" != 4 && "$sim_version" != 5 ]]; then
    echo "Unsupported SO101_SIM_VERSION=$sim_version; expected 4 or 5." >&2
    exit 1
fi
collector_pid_file="outputs/sim${sim_version}_slowed_1.7x_phase_governed_collector.pid"
collector_log="outputs/sim${sim_version}_slowed_1.7x_phase_governed_collector.log"
collector_unit="so101-bench-sim${sim_version}-slowed-1.7x-phase-governed.service"
if [[ ! -f "$collector_pid_file" ]]; then
    echo "No resumable slowed collector PID file exists: $collector_pid_file" >&2
    exit 1
fi

collector_pid="$(<"$collector_pid_file")"
if [[ ! "$collector_pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid collector PID file: $collector_pid_file" >&2
    exit 1
fi
if ! kill -0 "$collector_pid" 2>/dev/null; then
    echo "Collector PID $collector_pid is no longer running." >&2
    exit 1
fi

command_line="$(tr '\0' ' ' < "/proc/$collector_pid/cmdline")"
if [[ "$command_line" != *"scripts/so101_lerobot_collect_outcomes.py"* ]] || \
   [[ "$command_line" != *"so101_bench_sim_${sim_version}_slowed_1.7x_phase_governed"* ]]; then
    echo "Refusing to signal unexpected PID $collector_pid: $command_line" >&2
    exit 1
fi

kill -TERM "$collector_pid"
echo "Requested a graceful pause from PID $collector_pid."
echo "It will save the active episode, write an updated summary, and then exit."
echo "Progress log: $collector_log"
