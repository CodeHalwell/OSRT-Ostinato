#!/usr/bin/env bash
# Fire the whole ladder across Modal workspaces, one arm per workspace, detached.
#
#   scripts/launch_ladder.sh            # launch all six arms
#   scripts/launch_ladder.sh --dry-run  # print what would run, launch nothing
#   scripts/launch_ladder.sh a nohra    # just these arms
#
# Each workspace needs BOTH `hf-secret` (HF_TOKEN) and `wandb-secret`
# (WANDB_API_KEY) — the v6 names, already present in four workspaces.
# The script checks and refuses rather than burning a cold-start on a run that
# will die at the first HF call.
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
  [nohra]=gradio-winter-hack
  [g4]=build-small
)
ORDER=(a b c dense nohra g4)
STEPS="${STEPS:-8000}"

DRY=0; ARMS=()
for x in "$@"; do
  case "$x" in --dry-run) DRY=1 ;; *) ARMS+=("$x") ;; esac
done
[ ${#ARMS[@]} -eq 0 ] && ARMS=("${ORDER[@]}")

echo "ladder launch — ${#ARMS[@]} arm(s), ${STEPS} steps each (~$(python3 -c "print(round(${STEPS}*0.134/1000,2))")B tokens)"
echo

# 1. every target workspace must have the secret
declare -A CHECKED=()
for arm in "${ARMS[@]}"; do
  ws="${WS[$arm]:?unknown arm $arm}"
  [ -n "${CHECKED[$ws]:-}" ] && continue
  CHECKED[$ws]=1
  have=$(MODAL_PROFILE="$ws" uv run modal secret list 2>/dev/null)
  ok=1
  for sec in hf-secret wandb-secret; do
    if ! grep -q "$sec" <<<"$have"; then
      echo "  ✗ $ws is MISSING $sec"; ok=0; MISSING=1
    fi
  done
  [ "$ok" -eq 1 ] && echo "  ✓ $ws has hf-secret + wandb-secret"
done
[ -n "${MISSING:-}" ] && [ "$DRY" -eq 0 ] && { echo; echo "refusing to launch with missing secrets"; exit 1; }
echo

# 2. launch, detached
for arm in "${ARMS[@]}"; do
  ws="${WS[$arm]}"
  cmd=(uv run modal run --detach app.py --arm "$arm" --total-steps "$STEPS" --spawn)
  if [ "$DRY" -eq 1 ]; then
    echo "  [dry] MODAL_PROFILE=$ws ${cmd[*]}"
  else
    echo "  → $arm on $ws"
    MODAL_PROFILE="$ws" "${cmd[@]}" 2>&1 | grep -E "spawned|object_id|error|Error" || true
  fi
done
echo
echo "watch: W&B project 'osrt', runs osrt-v7-ladder-{a,b,c,dense,nohra,g4}"
echo "read:  roadmap §14.7 (G3a), §18.1 (E1), §17.2 (G4), §18.2 (E2 telemetry)"
