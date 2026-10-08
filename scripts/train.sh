#!/usr/bin/env bash
# Launch one reported run. The arm is the first argument:
#
#   cold             progressive exploration phase 1 (steps 1-25): every query
#                    forced, ordinary gated GRPO on the task reward, global
#                    token reduction, no support term. Saves global_step_25,
#                    which every arm below resumes.
#   main             phases 2-3 (steps 26-90) with COMET-GRPO. ALPHA=0 is the
#                    reported model; ALPHA=0.1|0.3|0.5 adds the efficiency
#                    reward (comet_gdpo + the cost reward manager).
#   wo_comet         ablation: the same schedule and support term, but ordinary
#                    GRPO -- one trajectory-level advantage on every router
#                    token, one global token mean.
#   wo_progressive   ablation: rho_b = 0 for every batch, from the base model.
#                    The support term (lambda_0 * rho_b) is inert too.
#   calibration      the rollout that freezes C_ref (Appendix E).
#
# Usage:
#   bash scripts/train.sh cold
#   ALPHA=0.1 bash scripts/train.sh main
# A run uses two GPUs; CUDA_VISIBLE_DEVICES picks which.
#   DRY_RUN=1 bash scripts/train.sh main        # resolve and validate only
# Extra arguments are forwarded to hydra verbatim.
set -euo pipefail
ARM=${1:-}; shift || true
source "$(dirname "${BASH_SOURCE[0]}")/paper_config.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOTAL_TRAINING_STEPS=90 SAVE_STEPS=${SAVE_STEPS:-50,75,90}
export TRAIN_FILES=$TRAIN_PARQUET
export ALLOW_WARM_START=1
export RESUME_MODE=${RESUME_MODE:-auto}

# Phases 2-3 continue the cold start: the run gets its own output dir whose
# global_step_25 is a symlink to the shared checkpoint, and a tracker that says
# 25 on a FIRST start only -- on a restart it keeps whatever the run last saved,
# so a relaunch never re-trains steps. This is a real resume (model, optimizer
# and dataloader state at row 800), so rho_b continues from batch_index 25.
resume_from_cold() {
    local cold=${COLD_CKPT:-$OUTPUTS/routeweaver_cold/checkpoints/global_step_25}
    [ -d "$cold/actor" ] || { echo "FATAL: cold-start checkpoint missing: $cold" >&2; exit 1; }
    mkdir -p "$OUTPUT_DIR/checkpoints"
    [ -e "$OUTPUT_DIR/checkpoints/global_step_25" ] || ln -s "$cold" "$OUTPUT_DIR/checkpoints/global_step_25"
    local track=$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt
    [ -f "$track" ] || echo 25 > "$track"
    export ROUTEWEAVER_SUPPORT=1      # lambda_b = SUPPORT_LAMBDA0 * rho_b, retires with forcing
}

case "$ARM" in
cold)
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-routeweaver_cold}
    export OUTPUT_DIR=$OUTPUTS/$EXPERIMENT_NAME
    export TRAIN_FILES=$COLD_PARQUET
    export RESUME_MODE=disable
    export TOTAL_TRAINING_STEPS=25 SAVE_STEPS=25
    export FORCED_START_BATCH=999                   # rho_b = 1 throughout
    export ADV_ESTIMATOR=grpo_gated ROUTEWEAVER_COMET=0 ROUTEWEAVER_LOSS_REDUCTION=global
    export ROUTEWEAVER_SUPPORT=0
    ;;
main)
    ALPHA=${ALPHA:-0}
    if [ "$ALPHA" = "0" ]; then
        export EXPERIMENT_NAME=${EXPERIMENT_NAME:-routeweaver_alpha0}
    else
        export EXPERIMENT_NAME=${EXPERIMENT_NAME:-routeweaver_alpha${ALPHA}}
    fi
    export OUTPUT_DIR=$OUTPUTS/$EXPERIMENT_NAME
    resume_from_cold
    export ROUTEWEAVER_COMET=1 ROUTEWEAVER_LOSS_REDUCTION=segment
    if [ "$ALPHA" = "0" ]; then
        export ADV_ESTIMATOR=comet_grpo
    else
        export ADV_ESTIMATOR=comet_gdpo COST_ALPHA=$ALPHA
        export C_REF=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['c_ref_usd'])" "$C_REF_FILE")
        export REWARD_MANAGER_NAME=CostRewardManager
        export REWARD_MANAGER_PATH=$REPO/rewards/cost_reward_manager.py
    fi
    ;;
wo_comet)
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ablation_wo_comet}
    export OUTPUT_DIR=$OUTPUTS/$EXPERIMENT_NAME
    resume_from_cold
    export ADV_ESTIMATOR=grpo_gated ROUTEWEAVER_COMET=0 ROUTEWEAVER_LOSS_REDUCTION=global
    ;;
wo_progressive)
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-ablation_wo_progressive}
    export OUTPUT_DIR=$OUTPUTS/$EXPERIMENT_NAME
    export FORCED_START_BATCH=-1 FORCED_ANNEAL_STEPS=1   # rho_b = 0 from batch 0
    export ADV_ESTIMATOR=comet_grpo ROUTEWEAVER_COMET=1 ROUTEWEAVER_LOSS_REDUCTION=segment
    export ROUTEWEAVER_SUPPORT=1                         # inert: lambda_b = lambda_0 * 0
    ;;
calibration)
    # 1024 rollouts (128 queries x 8) of the cold-start checkpoint with TRAINING
    # decoding, cost measured but never rewarded; then C_ref = P95 of the
    # per-trajectory worker cost. The frozen result is configs/cost_ref.json.
    export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cost_calib_128}
    export OUTPUT_DIR=$OUTPUTS/$EXPERIMENT_NAME
    CKPT=${COLD_CKPT:-$OUTPUTS/routeweaver_cold/checkpoints/global_step_25}
    HF=$OUTPUTS/hf_export/routeweaver_cold_step25
    # The rollout needs merged weights. Merging reads a sharded checkpoint and
    # writes ~8 GB, so it must not run just to resolve the configuration.
    if [ ! -f "$HF/model.safetensors" ] && ! ls "$HF"/model-0000*-of-*.safetensors >/dev/null 2>&1; then
        if [ "${DRY_RUN:-0}" = "1" ]; then
            # Nothing further can be resolved: the rollout runs against weights
            # that only exist once the merge has actually run.
            echo "DRY_RUN: would merge $CKPT/actor into $HF, then roll out"
            echo "         128 queries x 8 against it. Nothing launched."
            exit 0
        else
            "$PY" "$REPO/tools/merge_ckpt_to_hf.py" --ckpt "$CKPT/actor" --out "$HF" \
                || { echo "FATAL: merge failed" >&2; exit 1; }
        fi
    fi
    [ -f "$DATASET_ROOT/cost_calib/cost_calib_128.parquet" ] || {
        echo "FATAL: $DATASET_ROOT/cost_calib/cost_calib_128.parquet missing;" >&2
        echo "       run data/hf_download.py" >&2
        exit 1; }
    export MODEL_PATH=$HF VAL_FILES=$DATASET_ROOT/cost_calib/cost_calib_128.parquet
    export VAL_ONLY=True VAL_BEFORE_TRAIN=True TEST_FREQ=-1
    export VAL_N=8 VAL_DO_SAMPLE=True VAL_TEMPERATURE=0.8 VAL_TOP_P=0.95
    export RESUME_MODE=disable TOTAL_TRAINING_STEPS=1 SAVE_STEPS=99999
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} GPU_MEM_UTIL=0.55
    export DISPATCH_CONCURRENCY=20
    export ADV_ESTIMATOR=grpo_gated ROUTEWEAVER_COMET=0 ROUTEWEAVER_LOSS_REDUCTION=global
    export ROUTEWEAVER_SUPPORT=0 COST_ALPHA=0
    export REWARD_MANAGER_NAME=CostRewardManager
    export REWARD_MANAGER_PATH=$REPO/rewards/cost_reward_manager.py
    ;;
*)
    echo "usage: bash scripts/train.sh {cold|main|wo_comet|wo_progressive|calibration}" >&2
    exit 1
    ;;
esac

export WANDB_GROUP=${WANDB_GROUP:-routeweaver}
mkdir -p "$OUTPUT_DIR"
LOG=$OUTPUT_DIR/launch_$(date +%Y%m%d_%H%M%S).log
bash "$REPO/scripts/train_routeweaver.sh" "$@" > "$LOG" 2>&1
rc=$?
echo "$ARM: rc=$rc  log=$LOG"
if [ "$ARM" = calibration ] && [ "$rc" = 0 ] && [ "${DRY_RUN:-0}" != 1 ]; then
    "$PY" "$REPO/tools/cost_ref_from_dump.py" "$OUTPUT_DIR"
fi
exit $rc
