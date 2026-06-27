#!/usr/bin/env bash
# Run the harvester once per interval for a week, on a cluster.
#
# The harvester is one-shot (one search -> classify -> draft -> notify, then
# exit). This wrapper just fires it on a fixed interval for a bounded duration.
# The SQLite `seen` ledger guarantees each post is answered at most once across
# all runs, so re-running every hour only ever picks up genuinely new posts.
#
# Each iteration is isolated: if one run crashes (network blip, API error), it is
# logged and the loop continues to the next interval rather than dying.
#
# Usage (from the repo root):
#   nohup bash harvester/run_week.sh > harvester_week.out 2>&1 &
#
# Tunables via env (defaults in parens):
#   INTERVAL_SEC   seconds between runs               (3600 = hourly)
#   DURATION_SEC   total run window                   (604800 = 7 days)
#   MAX_GENS       --max-generations per run          (3)
#   HARVESTER_DB   sqlite ledger path                 (harvester_tracking_local.db)
#   EXTRA_ARGS     extra flags passed to orchestrate  (e.g. "--query 'israel gaza'")
set -uo pipefail

# Resolve the repo root from this script's location so cron/Slurm can call it by
# absolute path and still run uv in the right directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

INTERVAL_SEC="${INTERVAL_SEC:-3600}"
DURATION_SEC="${DURATION_SEC:-604800}"   # 7 days
MAX_GENS="${MAX_GENS:-3}"
export HARVESTER_DB="${HARVESTER_DB:-harvester_tracking_local.db}"

deadline=$(( $(date +%s) + DURATION_SEC ))
run=0

echo "[run_week] start: interval=${INTERVAL_SEC}s duration=${DURATION_SEC}s db=${HARVESTER_DB} max_gens=${MAX_GENS}"

while [ "$(date +%s)" -lt "$deadline" ]; do
  run=$(( run + 1 ))
  echo "===== [run_week] run #${run} at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="

  # A crash in one run must not kill the week; capture the status and keep going.
  if uv run python -m harvester.orchestrate --max-generations "$MAX_GENS" ${EXTRA_ARGS:-}; then
    echo "[run_week] run #${run} ok"
  else
    echo "[run_week] run #${run} FAILED (exit $?) — continuing"
  fi

  # Don't oversleep past the deadline on the final iteration.
  now=$(date +%s)
  [ "$now" -ge "$deadline" ] && break
  remaining=$(( deadline - now ))
  sleep "$(( INTERVAL_SEC < remaining ? INTERVAL_SEC : remaining ))"
done

echo "[run_week] done after ${run} runs at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
