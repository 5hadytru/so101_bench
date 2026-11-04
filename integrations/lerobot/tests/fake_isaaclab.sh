#!/usr/bin/env bash
# Stands in for isaaclab.sh: `fake_isaaclab.sh -p <server> <args>` runs the fake server instead.
shift 2
exec "${FAKE_PYTHON:-python3}" "$(dirname "$0")/fake_sim_server.py" "$@"
