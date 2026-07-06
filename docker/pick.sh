#!/bin/bash
# Interactive object picker + GR00T eval launcher for SO-101 Bench.
#
#   ./docker/pick.sh
#
# Lists all benchmark objects, lets you choose one or more, builds a task file,
# and runs the eval (in Docker via so101.sh by default; set NATIVE=1 to run with
# ~/IsaacLab/isaaclab.sh instead). The GR00T policy server must be up on :5555.
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT_DIR"

# --- Object catalog (mirrors OBJECT_SPLITS in benchmark.py) ---
SEEN=("black glasses" "silver glasses" "white pen" "black pen" "altoids container" \
      "blue pliers" "green clip" "pink eraser" "yellow wires" "grey wires" \
      "black screwdriver" "yellow screwdriver" "red tape" "black tape" "cardboard box" \
      "flower pot" "cooking spoon" "yellow toy car" "grey toy car" "green shoes" \
      "black shoes" "blue bowl" "blue scissors")
UNSEEN_SC=("white glasses" "blue clip" "blue tape" "yellow tape" "blue screwdriver" \
           "pink bowl" "white bowl" "black wires" "brown wires" "orange toy car" \
           "blue pen" "red pen" "white shoes")
UNSEEN_UC=("blue highlighter" "purple toothbrush" "blue controller" "action figure" \
           "razor" "silver tongs" "playing cards" "candy bar" "toy fire truck" \
           "toy monster truck" "toy dinosaur" "sponge" "yellow flashlight")
ALL=("${SEEN[@]}" "${UNSEEN_SC[@]}" "${UNSEEN_UC[@]}")

# --- Print the menu ---
printf '\n\033[1mSO-101 Bench — pick object(s) for GR00T eval\033[0m\n'
i=0
printf '\n\033[32m== Seen (fine-tuned, best success) ==\033[0m\n'
for o in "${SEEN[@]}";      do i=$((i+1)); printf '%3d) %s\n' "$i" "$o"; done
printf '\n\033[33m== Unseen / seen-class ==\033[0m\n'
for o in "${UNSEEN_SC[@]}"; do i=$((i+1)); printf '%3d) %s\n' "$i" "$o"; done
printf '\n\033[31m== Unseen / unseen-class ==\033[0m\n'
for o in "${UNSEEN_UC[@]}"; do i=$((i+1)); printf '%3d) %s\n' "$i" "$o"; done

# --- Read selection ---
printf '\nEnter object number(s), space/comma separated (e.g. "4" or "4 13 8 23"): '
read -r picks
picks=${picks//,/ }
SEL=()
for n in $picks; do
  if ! [[ "$n" =~ ^[0-9]+$ ]] || (( n < 1 || n > ${#ALL[@]} )); then
    echo "Invalid selection: '$n' (must be 1-${#ALL[@]})" >&2; exit 1
  fi
  SEL+=("${ALL[$((n-1))]}")
done
[ ${#SEL[@]} -eq 0 ] && { echo "No objects selected." >&2; exit 1; }

# --- Choose task family ---
printf '\nTask:  1) bin (grasp into bin)   2) next-to   3) between   4) move   [1]: '
read -r t; t=${t:-1}

count=${#SEL[@]}
_json_objs() { printf '"%s"' "${SEL[0]}"; for o in "${SEL[@]:1}"; do printf ', "%s"' "$o"; done; }

case "$t" in
  1) TASK_ID="So101Bench-Bin-v0"
     [[ "$count" != 1 && "$count" != 4 ]] && { echo "Bin needs 1 or 4 objects (you picked $count)." >&2; exit 1; }
     INSTR="Place each object in the plastic bin" ;;
  2) TASK_ID="So101Bench-NextTo-v0"
     [ "$count" -ne 4 ] && { echo "Next-to needs exactly 4 objects." >&2; exit 1; }
     INSTR="Place the ${SEL[0]} next to the ${SEL[1]}" ;;
  3) TASK_ID="So101Bench-Between-v0"
     [ "$count" -ne 4 ] && { echo "Between needs exactly 4 objects." >&2; exit 1; }
     INSTR="Place the ${SEL[0]} between the ${SEL[1]} and the ${SEL[2]}" ;;
  4) TASK_ID="So101Bench-Move-v0"
     [ "$count" -ne 4 ] && { echo "Move needs exactly 4 objects." >&2; exit 1; }
     printf 'Direction (left/right/forward/backward) [forward]: '; read -r d; d=${d:-forward}
     INSTR="Move the ${SEL[0]} ${d}" ;;
  *) echo "Unknown task '$t'." >&2; exit 1 ;;
esac

# --- Write the task file ---
OUT="tasks/picker.jsonl"
printf '{"objects": [%s], "instruction": "%s"}\n' "$(_json_objs)" "$INSTR" > "$OUT"
echo
echo "Wrote $OUT:"
cat "$OUT"

# --- Launch ---
REPO="data/lerobot/picker_$(echo "${SEL[0]}" | tr ' ' '_')"
echo
echo "Launching eval  (task=$TASK_ID, repo_root=$REPO) ..."
if [ -n "${DRY_RUN:-}" ]; then
  echo "[dry-run] TASK=$TASK_ID EPISODES=$OUT NUM_EPISODES=1 REPO_ROOT=$REPO ./docker/so101.sh eval"
  exit 0
fi
if [ -n "${NATIVE:-}" ]; then
  exec ~/IsaacLab/isaaclab.sh -p scripts/groot_eval.py \
    --task "$TASK_ID" --episodes_jsonl "$OUT" \
    --policy_host localhost --policy_port 5555 \
    --action_horizon 16 --use_overhead_init true \
    --num_episodes 1 --record_dataset --repo_root "$REPO" "$@"
else
  TASK="$TASK_ID" EPISODES="$OUT" NUM_EPISODES=1 REPO_ROOT="$REPO" \
    exec ./docker/so101.sh eval "$@"
fi
