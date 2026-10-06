#!/usr/bin/env bash
# Resumable end-to-end pipeline for the remaining compute. Each stage is skipped when its
# completion marker exists, and every stage is itself resumable (completed runs, folds and
# feature checkpoints are reused), so rerunning after a crash or suspend continues the work.
#
#   setsid nohup scripts/run_pipeline.sh > results/logs/pipeline.log 2>&1 < /dev/null &
# The whole script is one function, parsed in full before it runs, so editing this file
# while a pipeline is running cannot corrupt the running instance.
main() {
  set -euo pipefail
  cd "$(dirname "$0")/.."

  # This laptop hangs when it resumes from suspend, which kills long jobs. While the pipeline
  # runs, block suspend from idle, the lid switch and the suspend key (released on exit).
  if [[ -z "${SPERMTRIAGE_INHIBITED:-}" ]] && command -v systemd-inhibit > /dev/null; then
    export SPERMTRIAGE_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch:handle-suspend-key \
      --who=spermtriage --why="research pipeline running" --mode=block "$0" "$@"
  fi
  PY=.venv/bin/python
  MARK=results/logs/pipeline_markers
  mkdir -p "$MARK"

  stage() {  # stage <name> <command...>
    local name=$1; shift
    if [[ -f "$MARK/$name.done" ]]; then echo "[skip] $name"; return; fi
    echo "[start] $name $(date -Is)"
    "$@"
    touch "$MARK/$name.done"
    echo "[done] $name $(date -Is)"
  }

  stage ft_smids_folds   .venv/bin/spermtriage train --dataset smids --model mobilenetv3-ft
  stage method_features  $PY -m spermtriage.method.run_study features
  stage method_inner     $PY -m spermtriage.method.run_study inner --tag v1
  stage method_main      $PY -m spermtriage.method.run_study main --tag v1 --repeats 5
  stage method_ablations $PY -m spermtriage.method.run_study ablations --tag v1
  stage method_invar     $PY -m spermtriage.method.run_study invariance --tag v1
  stage method_effic     $PY -m spermtriage.method.run_study efficiency --tag v1
  stage method_ceiling   $PY -m spermtriage.method.run_study ceiling --tag v1
  stage method_tables    $PY -m spermtriage.method.run_study tables --tag v1
  echo "[pipeline complete] $(date -Is)"
}

main "$@"
exit
