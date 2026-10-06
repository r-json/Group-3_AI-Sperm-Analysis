#!/usr/bin/env bash
# Compact view of the research pipeline's progress.
#   scripts/progress.sh              print once
#   watch -n 30 scripts/progress.sh  refresh every 30 s (Ctrl+C to quit)
cd "$(dirname "$0")/.."
LOG=results/logs/pipeline.log
MARK=results/logs/pipeline_markers
STAGES=(ft_smids_folds method_features method_inner method_main method_ablations method_invar method_effic method_ceiling method_tables)

echo "SpermTriage pipeline - $(date '+%Y-%m-%d %H:%M:%S')"
if pgrep -f run_pipeline.sh > /dev/null; then echo "status: RUNNING"; else echo "status: NOT RUNNING (restart: setsid nohup scripts/run_pipeline.sh >> $LOG 2>&1 < /dev/null &)"; fi
echo
for s in "${STAGES[@]}"; do
  if [[ -f "$MARK/$s.done" ]]; then echo "  [x] $s"
  elif grep -q "\[start\] $s" "$LOG" 2>/dev/null; then echo "  [>] $s   <- running"
  else echo "  [ ] $s"; fi
done
echo
echo "SMIDS fine-tune folds:"
for k in 0 1 2 3 4; do
  f=results/runs/main/smids/mobilenetv3-ft/fold$k/provenance.json
  [[ -f $f ]] && echo "  fold $k: done" || echo "  fold $k: pending"
done
echo
echo "Latest activity:"
grep -E "finetune epoch|warmup epoch|Early stop|=== main|canon: |d4: |inner \{|Traceback|Error" "$LOG" 2>/dev/null \
  | tail -4 | sed -E 's/^[0-9-]+ //; s/,[0-9]+ INFO [a-z._]+: / /' | cut -c1-110
echo
echo "Method runs logged (EXPERIMENTS.md):"
grep -E "^\| 20" EXPERIMENTS.md | tail -4 | cut -d'|' -f3-9 | cut -c1-110
echo
printf "CPU: %s MHz, load %s\n" "$(awk -F: '/MHz/{printf "%.0f", $2; exit}' /proc/cpuinfo)" "$(cut -d' ' -f1-3 /proc/loadavg)"
