#!/usr/bin/env bash
# Fire the whole ladder across Modal workspaces, one arm per workspace, detached.
#
#   scripts/launch_ladder.sh            # launch all six arms
#   scripts/launch_ladder.sh --dry-run  # print what would run, launch nothing
#   scripts/launch_ladder.sh a hra      # just these arms
#
# Arms: a b c dense hra g4 (osrt.presets.LADDER_ARMS). `hra` is the E1
# counter-arm (roadmap §18.1): arm a with the per-loop HRA adapters on.
#
# Each workspace needs BOTH `hf-secret` (HF_TOKEN) and `wandb-secret`
# (WANDB_API_KEY) — the v6 names, already present in four workspaces.
# The script checks and refuses rather than burning a cold-start on a run that
# will die at the first HF call. Exits 1 if any arm failed to launch.
set -euo pipefail
cd "$(dirname "$0")/.."

# arm -> workspace. Six arms over the FOUR workspaces that carry both
# hf-secret and wandb-secret (agents-of-output has neither — add them there
# to spread wider). The two G3a controls share; the experiments each get one.
declare -A WS=(
  [a]=danielhalwell
  [b]=build-small
  [c]=codhe-hugging-mcp
  [dense]=danielhalwell
  [hra]=gradio-winter-hack
  [g4]=build-small
)
ORDER=(a b c dense hra g4)
STEPS="${STEPS:-8000}"

DRY=0; ARMS=()
for x in "$@"; do
  case "$x" in --dry-run) DRY=1 ;; *) ARMS+=("$x") ;; esac
done
[ ${#ARMS[@]} -eq 0 ] && ARMS=("${ORDER[@]}")

echo "ladder launch — ${#ARMS[@]} arm(s), ${STEPS} steps each (~$(python3 -c "print(round(${STEPS}*0.134/1000,2))")B tokens)"
echo

# 1. every target workspace must have the secret. Exact name match on the
#    JSON listing (`hf-secret-old` must not pass for `hf-secret`), and a
#    failing `modal secret list` (bad profile, expired token) prints its own
#    error instead of masquerading as "missing secret".
declare -A CHECKED=()
for arm in "${ARMS[@]}"; do
  ws="${WS[$arm]:?unknown arm $arm}"
  [ -n "${CHECKED[$ws]:-}" ] && continue
  CHECKED[$ws]=1
  if ! have=$(MODAL_PROFILE="$ws" uv run modal secret list --json); then
    echo "  ✗ $ws: 'modal secret list' failed (see above)"; MISSING=1; continue
  fi
  names=$(python3 -c 'import json,sys; print("\n".join(r["Name"] for r in json.load(sys.stdin)))' <<<"$have") \
    || { echo "  ✗ $ws: could not parse the secret listing"; MISSING=1; continue; }
  ok=1
  for sec in hf-secret wandb-secret; do
    if ! grep -qxF "$sec" <<<"$names"; then
      echo "  ✗ $ws is MISSING $sec"; ok=0; MISSING=1
    fi
  done
  [ "$ok" -eq 1 ] && echo "  ✓ $ws has hf-secret + wandb-secret"
done
[ -n "${MISSING:-}" ] && [ "$DRY" -eq 0 ] && { echo; echo "refusing to launch with missing secrets"; exit 1; }
echo

# 2. launch, detached. The grep keeps the log readable; the status we judge
#    is modal's own (PIPESTATUS[0]), never grep's.
declare -A RESULT=()
for arm in "${ARMS[@]}"; do
  ws="${WS[$arm]}"
  cmd=(uv run modal run --detach app.py --arm "$arm" --total-steps "$STEPS" --spawn)
  if [ "$DRY" -eq 1 ]; then
    echo "  [dry] MODAL_PROFILE=$ws ${cmd[*]}"
    continue
  fi
  echo "  → $arm on $ws"
  set +e
  MODAL_PROFILE="$ws" "${cmd[@]}" 2>&1 | grep -E "spawned|object_id|error|Error|Traceback"
  RESULT[$arm]=${PIPESTATUS[0]}
  set -e
done
echo

# 3. per-arm verdict; exit 1 if any launch failed
FAILED=0
if [ "$DRY" -eq 0 ]; then
  echo "summary:"
  for arm in "${ARMS[@]}"; do
    if [ "${RESULT[$arm]}" -eq 0 ]; then
      echo "  OK    $arm  (${WS[$arm]})"
    else
      echo "  FAIL  $arm  (${WS[$arm]})  modal exit ${RESULT[$arm]}"; FAILED=1
    fi
  done
  echo
fi
echo "watch: W&B project 'osrt', runs osrt-v7-ladder-{a,b,c,dense,hra,g4}"
echo "read:  roadmap §14.7 (G3a), §18.1 (E1), §17.2 (G4), §18.2 (E2 telemetry)"
[ "$FAILED" -eq 0 ] || exit 1
