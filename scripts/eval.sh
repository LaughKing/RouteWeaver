#!/usr/bin/env bash
# Evaluate one checkpoint. Greedy router decoding, one system rollout per
# question, workers at temperature 0.2 (Appendix F) -- this is the training
# launcher in VAL_ONLY mode, so evaluation and training share one code path.
#
#   bash scripts/eval.sh paper          CKPT=... TAG=... GPU=6 [OOD=1]
#       the nine in-distribution benchmarks (1,060 questions) and, with OOD=1,
#       the held-out ones; then prints the paper's table.
#
#   bash scripts/eval.sh one            CKPT=... VAL=... NAME=... GPU=6 [COST=1]
#       one checkpoint on one parquet. COST=1 scores with the cost reward
#       manager at alpha 0 -- the same verdicts, plus the per-call worker-cost
#       columns the cost analyses use.
set -uo pipefail
MODE=${1:-paper}; shift || true
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paper_config.sh"

eval_one() {   # CKPT VAL NAME GPU [COST]
  : "${CKPT:?}" "${VAL:?}" "${NAME:?}" "${GPU:?}"
  VAL=$(realpath "$VAL"); CKPT=$(realpath "$CKPT")
  # CKPT may be an already-merged HF directory (a Hub snapshot, or an earlier
  # export) or a training checkpoint, which is FSDP-sharded and merged once.
  if [ -f "$CKPT/config.json" ]; then
      HF=${HF:-$CKPT}
  else
      HF=${HF:-$OUTPUTS/hf_export/$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT")}
      if [ ! -f "$HF/model.safetensors" ] && ! ls "$HF"/model-0000*-of-*.safetensors >/dev/null 2>&1; then
          "$PY" "$REPO/tools/merge_ckpt_to_hf.py" --ckpt "$CKPT/actor" --out "$HF" \
              || { echo "FATAL: merge failed" >&2; exit 1; }
      fi
  fi

  OUT=$OUTPUTS/eval/$NAME
  [ -e "$OUT" ] && { echo "FATAL: $OUT exists" >&2; exit 1; }
  mkdir -p "$OUT"
  (
    export EXPERIMENT_NAME=$NAME OUTPUT_DIR=$OUT MODEL_PATH=$HF
    export WANDB_GROUP=routeweaver-eval WANDB_MODE=offline
    export CUDA_VISIBLE_DEVICES=$GPU GPU_MEM_UTIL=0.55
    export TRAIN_FILES=$TRAIN_PARQUET VAL_FILES=$VAL
    export VAL_ONLY=True VAL_BEFORE_TRAIN=True TEST_FREQ=-1
    export VAL_N=1 VAL_DO_SAMPLE=False VAL_TEMPERATURE=0 VAL_TOP_P=1.0
    export RESUME_MODE=disable TOTAL_TRAINING_STEPS=1 SAVE_STEPS=99999 ALLOW_WARM_START=1
    export DISPATCH_CONCURRENCY=${CONC:-20}
    export ROUTEWEAVER_SUPPORT=0 ROUTEWEAVER_COMET=0 ADV_ESTIMATOR=grpo_gated ROUTEWEAVER_LOSS_REDUCTION=global
    if [ "${COST:-0}" = "1" ]; then
      export COST_ALPHA=0 C_REF=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['c_ref_usd'])" "$C_REF_FILE")
      export REWARD_MANAGER_NAME=CostRewardManager
      export REWARD_MANAGER_PATH=$REPO/rewards/cost_reward_manager.py
    fi
    # the dataset check only validates TRAIN_FILES, which VAL_ONLY never trains on
    exec bash "$REPO/scripts/train_routeweaver.sh"
  ) > "$OUT/eval.out" 2>&1
  rc=$?
  echo "eval $NAME rc=$rc"
  return $rc
}

case "$MODE" in
one)
  eval_one
  exit $?
  ;;
paper)
  : "${CKPT:?}" "${TAG:?}" "${GPU:?}"
  E=$DATASET_ROOT/eval
  # A failed set aborts the sweep: a table summarized over a missing dump
  # would be a table of fewer benchmarks wearing the same name.
  run() { CKPT=$CKPT VAL=$1 NAME=$2 GPU=$GPU COST=${COST:-0} eval_one || exit $?; }
  run $E/eval_600_free.parquet               eval-$TAG-set600
  run $E/eval_700_free.parquet               eval-$TAG-set700
  run $E/eval_triviaqa120_free.parquet       eval-$TAG-triviaqa120
  if [ "${OOD:-0}" = "1" ]; then
      for s in aime149 gpqa_diamond198 bbh1080; do
          COST=1 run $E/ood_${s}_free.parquet eval-$TAG-ood-$s
      done
  fi
  "$PY" "$REPO/evaluation/summarize_paper.py" "$OUTPUTS/eval" "$TAG"
  ;;
*)
  echo "usage: bash scripts/eval.sh {paper|one} (see the header for variables)" >&2
  exit 1
  ;;
esac
