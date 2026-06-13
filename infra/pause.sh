#!/usr/bin/env bash
# Cost kill-switch: stop the harvester from spending WITHOUT tearing down the
# stack (so you can resume later). Budgets only *alert*; this actually halts.
#
#   infra/pause.sh           # pause: disable schedule, stop running tasks
#   infra/pause.sh resume    # re-enable the hourly schedule
#   infra/pause.sh status    # show schedule state + any running tasks + MTD cost
#
# For a FULL stop of all spend (incl. the ~$0.05/hr NAT), use `terraform destroy`
# (see infra/DEPLOY.md teardown) — this script leaves the VPC/NAT up.
set -euo pipefail

REGION="${AWS_REGION:-eu-west-1}"
CLUSTER="cmv-harvester-cluster"
SCHEDULE="cmv-harvester-harvester"
ACTION="${1:-pause}"

sched_state() { aws scheduler get-schedule --name "$SCHEDULE" --region "$REGION" --query State --output text; }

set_sched() {
  local state="$1"
  aws scheduler get-schedule --name "$SCHEDULE" --region "$REGION" > /tmp/_sched.json
  aws scheduler update-schedule --name "$SCHEDULE" --region "$REGION" \
    --state "$state" \
    --schedule-expression "$(python3 -c "import json;print(json.load(open('/tmp/_sched.json'))['ScheduleExpression'])")" \
    --flexible-time-window '{"Mode":"OFF"}' \
    --target "$(python3 -c "import json;print(json.dumps(json.load(open('/tmp/_sched.json'))['Target']))")" \
    >/dev/null
}

stop_running_tasks() {
  for t in $(aws ecs list-tasks --cluster "$CLUSTER" --region "$REGION" --desired-status RUNNING --query "taskArns[]" --output text); do
    echo "  stopping task $t"
    aws ecs stop-task --cluster "$CLUSTER" --task "$t" --region "$REGION" --reason "cost-guard pause" >/dev/null
  done
}

case "$ACTION" in
  pause)
    echo "[pause] disabling schedule…"; set_sched DISABLED
    echo "[pause] stopping any running tasks…"; stop_running_tasks
    echo "[pause] done. schedule=$(sched_state). NAT still up — terraform destroy for full stop."
    ;;
  resume)
    echo "[resume] enabling schedule…"; set_sched ENABLED
    echo "[resume] done. schedule=$(sched_state)."
    ;;
  status)
    echo "schedule: $(sched_state)"
    echo "running tasks: $(aws ecs list-tasks --cluster "$CLUSTER" --region "$REGION" --desired-status RUNNING --query "length(taskArns)" --output text)"
    echo -n "month-to-date cost (USD): "
    aws ce get-cost-and-usage --time-period Start="$(date -u +%Y-%m-01)",End="$(date -u +%Y-%m-%d)" \
      --granularity MONTHLY --metrics UnblendedCost \
      --query "ResultsByTime[0].Total.UnblendedCost.Amount" --output text 2>/dev/null || echo "(Cost Explorer unavailable)"
    ;;
  *)
    echo "usage: $0 [pause|resume|status]" >&2; exit 2 ;;
esac
