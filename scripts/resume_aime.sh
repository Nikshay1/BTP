#!/usr/bin/env bash
# Finish the AIME tuning check, then (only if voting is not worse than the single agent on the
# tuning problems) the AIME test, retrying every 30 minutes while Bedrock's daily token quota is
# spent (the runner exits 5 then and leaves interrupted runs unrecorded). Log: runs/resume-aime.log
# Stop it with: pkill -f resume_aime.sh
set -u
cd "$(dirname "$0")/.."
log=runs/resume-aime.log

run_until_done() {  # config, max-usd
  while true; do
    echo "$(date -u +%FT%TZ) start $1" >>"$log"
    uv run bioalloc experiment run "$1" --max-usd "$2" \
      --accept-code-change "driver, scripts and docs only" >>"$log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) exit $rc for $1" >>"$log"
    case $rc in
      0) return 0 ;;
      5) sleep 1800 ;;
      *) return $rc ;;
    esac
  done
}

run_until_done configs/aime-tune-v2.json 12 || exit 1
# Gate written before the tuning check finished (R23): majority5 must solve at least as many
# tuning problems as the single agent did (21/30).
ok=$(python3 - <<'EOF'
import json
rows = [json.loads(x) for x in open("runs/aime-tune-v2/results.jsonl") if x.strip()]
print(int(sum(r["correct"] for r in rows if r["system"] == "majority5") >= 21))
EOF
)
echo "$(date -u +%FT%TZ) gate majority5>=21: $ok" >>"$log"
[ "$ok" = 1 ] || exit 0
run_until_done configs/aime-test.json 55
echo "$(date -u +%FT%TZ) all done" >>"$log"
