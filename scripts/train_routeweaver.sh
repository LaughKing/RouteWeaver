#!/usr/bin/env bash
# RouteWeaver training / evaluation entry point (verl 0.8, pip-installed).
#
# Normally launched through scripts/train.sh / scripts/eval.sh, which set
# the paper configuration (scripts/paper_config.sh).
# Usage:
#   bash scripts/train.sh cold                    # go through the dispatcher
#   DRY_RUN=1 bash scripts/train_routeweaver.sh   # print resolved config only
# Extra arguments are forwarded to hydra verbatim.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_DIRS=$REPO/agent_loops:$REPO/trainers:$REPO/rewards:$REPO/routing:$REPO/evaluation:$REPO/evaluation/judges
DATA=$REPO/data
DATASET_ROOT=${ROUTEWEAVER_DATASETS:-$REPO/datasets}

# ---- data / model -----------------------------------------------------------
TRAIN_FILES=${TRAIN_FILES:-$DATASET_ROOT/parquet/routeweaver_train.parquet}
VAL_FILES=${VAL_FILES:-$DATASET_ROOT/parquet/val_stub.parquet}
# Router policy: local weights. Workers are frozen HTTP endpoints.
MODEL_PATH=${MODEL_PATH:-${ROUTER_MODEL:-$HOME/models/Qwen3-4B-Instruct-2507}}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-routeweaver}
PROJECT_NAME=${PROJECT_NAME:-routeweaver}
WANDB_GROUP=${WANDB_GROUP:-routeweaver}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/outputs/$EXPERIMENT_NAME}
# One JSONL per launch attempt.
VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-$OUTPUT_DIR/metrics_$(date +%Y%m%d_%H%M%S).jsonl}

# ---- schedule ---------------------------------------------------------------
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# Explicit checkpoint step list (enforced by checkpoint_steps.py).
SAVE_STEPS=${SAVE_STEPS:-10,25,50}
# true = an illegal action gets a rules cue and a retry; false = it ends the rollout.
POLICY_VIOLATION_RECOVERY=${POLICY_VIOLATION_RECOVERY:-false}
# Routing-state header before every agentic action.
ROUTING_STATE=${ROUTING_STATE:-true}
TEST_FREQ=${TEST_FREQ:--1}            # no validation in the baseline
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
# Zero-update evaluation: set VAL_ONLY and VAL_BEFORE_TRAIN together.
VAL_ONLY=${VAL_ONLY:-False}
RESUME_MODE=${RESUME_MODE:-auto}      # auto within THIS run only; the output dir is separate
# The agent loop that drives the rollout (the only one registered).
AGENT_LOOP=${AGENT_LOOP:-routeweaver}
AGENT_LOOP_CONFIG=${AGENT_LOOP_CONFIG:-$REPO/configs/agent_loop.yaml}
# alpha_t = 1 - (batch_index - START)/STEPS, read by routeweaver only.
FORCED_START_BATCH=${FORCED_START_BATCH:-25}
FORCED_ANNEAL_STEPS=${FORCED_ANNEAL_STEPS:-50}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:--1}    # -1 = keep every one

# ---- GRPO -------------------------------------------------------------------
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}      # prompts per step
ROLLOUT_N=${ROLLOUT_N:-8}                     # K per prompt -> 256 trajectories
# Scaled by rollout.n in verl: one on-policy update per step.
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MICRO_BATCH_PER_GPU=${MICRO_BATCH_PER_GPU:-1}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo_gated}
DATA_SEED=${DATA_SEED:-42}
# Must stay False: row order encodes the per-step mode quota.
DATA_SHUFFLE=${DATA_SHUFFLE:-False}

# ---- lengths ----------------------------------------------------------------
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-6144}
# Covers router turns plus injected observations.
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_OBS_LENGTH=${MAX_OBS_LENGTH:-2048}        # per TURN, split across a layer
# 0.30 avoids CUDA OOM in update_actor.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30}
# Fused linear + log-softmax: the full logits tensor is never materialised.
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}

# ---- environment (frozen workers) ------------------------------------------
CHANNEL_CONFIG=${CHANNEL_CONFIG:-$REPO/configs/route_channels.yaml}
WORKER_IDS=${WORKER_IDS:-worker_1,worker_2,worker_3,worker_4,worker_5,worker_6}
WORKER_TEMPERATURE=${WORKER_TEMPERATURE:-0.2}
WORKER_MAX_TOKENS=${WORKER_MAX_TOKENS:-4096}
SELECTOR_MAX_TOKENS=${SELECTOR_MAX_TOKENS:-64}
MAX_WORKERS_V2=${MAX_WORKERS_V2:-6}
# Global worker concurrency bound; keep low for rate-limited providers.
DISPATCH_CONCURRENCY=${DISPATCH_CONCURRENCY:-20}
AGENT_LOOP_WORKERS=${AGENT_LOOP_WORKERS:-1}

# ---- router sampling --------------------------------------------------------
ROUTER_TEMPERATURE=${ROUTER_TEMPERATURE:-0.8}
ROUTER_TOP_P=${ROUTER_TOP_P:-0.95}
# -1 = no top-k (overrides the model's generation_config).
ROUTER_TOP_K=${ROUTER_TOP_K:--1}

# 4 GiB Ray object store.
OBJECT_STORE_BYTES=${OBJECT_STORE_BYTES:-4294967296}
# Minimum free host RAM before launching.
MIN_FREE_RAM_GB=${MIN_FREE_RAM_GB:-130}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
N_GPUS=$(awk -F, '{print NF}' <<< "$CUDA_DEVICES")
# Loggers: console, wandb, local JSONL.
LOGGER=${LOGGER:-'[console,wandb,file]'}
PY=${PY:-${PY:-python}}

# ---- fail fast --------------------------------------------------------------
for f in "$TRAIN_FILES" "$VAL_FILES" "$CHANNEL_CONFIG"; do
    [ -f "$f" ] || { echo "FATAL: not found: $f" >&2; exit 1; }
done
[ -d "$MODEL_PATH" ] || { echo "FATAL: model not found: $MODEL_PATH" >&2; exit 1; }
FREE_RAM_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${FREE_RAM_GB:-0}" -lt "$MIN_FREE_RAM_GB" ]; then
    echo "FATAL: only ${FREE_RAM_GB}GB host RAM available, need ${MIN_FREE_RAM_GB}GB." >&2
    echo "       This node is shared; Ray kills workers mid-rollout below that." >&2
    ps -eo rss,user,args --sort=-rss | head -4 | awk '{printf "       %6.0fGB %s %s\n", $1/1048576, $2, $3}' >&2
    exit 1
fi
[ -f "$MODEL_PATH/config.json" ] || { echo "FATAL: $MODEL_PATH is not a local HF model" >&2; exit 1; }
[ "$DATA_SHUFFLE" = "False" ] || { echo "FATAL: DATA_SHUFFLE must be False -- the parquet row order IS the 7/7/7/6/5 quota" >&2; exit 1; }
[ "$TRAIN_BATCH_SIZE" = "32" ] || echo "WARN: train_batch_size != 32 breaks the per-batch domain quota"

# The parquet must match the current mode_selector prompt.
"$PY" - "$REPO" "$TRAIN_FILES" "${EXPECT_MODE_QUOTA:-11,11,10}" <<'PYCHECK' || exit 1
import json, sys
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "routing"))
import pandas as pd
import mode_selector
meta_path = Path(sys.argv[2] + ".meta.json")
if not meta_path.exists():
    sys.exit(f"FATAL: {meta_path} missing; the parquet needs its sidecar")
meta = json.loads(meta_path.read_text())
if meta["selector_version"] != mode_selector.SELECTOR_VERSION:
    sys.exit(f"FATAL: parquet was baked with selector {meta['selector_version']} "
             f"but mode_selector is now {mode_selector.SELECTOR_VERSION}. Rebuild.")
frame = pd.read_parquet(sys.argv[2], columns=["prompt", "extra_info"])
row = frame.iloc[0]
question = json.loads(row["extra_info"]["payload_json"])["raw_question"]
if row["prompt"][0]["content"] != mode_selector.build(question):
    sys.exit("FATAL: baked prompt != mode_selector.build(question). Rebuild.")
counts, modes = {}, {}
for info in frame["extra_info"][:32]:
    counts[info["domain"]] = counts.get(info["domain"], 0) + 1
    forced = info.get("forced_mode")
    if forced not in ("single", "multi", "agentic"):
        sys.exit(f"FATAL: row without a legal forced_mode: {forced!r}")
    modes[forced] = modes.get(forced, 0) + 1
want = {"MATH": 7, "CODE": 7, "KNOWLEDGE": 7, "REASON": 6, "RECALL": 5}
if counts != want:
    sys.exit(f"FATAL: first 32 rows are {counts}, not {want}")
# Expected sorted per-step mode counts, e.g. "11,11,10" or "32".
expect = sorted(int(x) for x in sys.argv[3].split(",") if x)
if sorted(modes.values()) != expect:
    sys.exit(f"FATAL: first step mode quota {modes}, expected {expect}")
print(f"mode quota step 0: {modes}")
print(f"dataset check OK: {len(frame)} rows, selector {meta['selector_version']}, "
      f"batch quota {counts}")
PYCHECK

# ---- the starting weights must be the UNTRAINED base, not a warm checkpoint --
if [ "${ALLOW_WARM_START:-0}" != "1" ]; then
  EXISTING=$(ls -d "$OUTPUT_DIR"/checkpoints/global_step_* 2>/dev/null | head -1 || true)
  if [ -n "$EXISTING" ]; then
    echo "FATAL: $OUTPUT_DIR/checkpoints already holds $EXISTING." >&2
    echo "       This run must start from untrained weights. Use a fresh" >&2
    echo "       OUTPUT_DIR, or set ALLOW_WARM_START=1 deliberately." >&2
    exit 1
  fi
fi
echo "start weights   : $MODEL_PATH  (global_step ${INIT_GLOBAL_STEP:-0}, untrained)"
"$PY" - "$MODEL_PATH" <<'PYWEIGHTS' || exit 1
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
shards = sorted(p.name for p in d.glob("*.safetensors"))
assert shards, f"no safetensors in {d}"
assert not list(d.glob("**/optim_world_size_*")),     f"{d} carries optimizer state -- that is a trained checkpoint, not base weights"
print(f"  verified base weights: {cfg.get('architectures')} "
      f"hidden={cfg.get('hidden_size')} shards={len(shards)}")
PYWEIGHTS

PIDFILE=${PIDFILE:-$OUTPUT_DIR/trainer.pid}
mkdir -p "$OUTPUT_DIR"
if [ -f "$PIDFILE" ]; then
    OLD=$(cat "$PIDFILE" 2>/dev/null || echo "")
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "FATAL: a trainer for $EXPERIMENT_NAME is already running (pid $OLD)." >&2
        echo "       Two trainers would share one checkpoint dir and corrupt it." >&2
        exit 1
    fi
    echo "note: stale pidfile for pid ${OLD:-?}; taking over"
fi


# ---- resume ----------------------------------------------------------------
RESUME_STEP=0
if [ "$RESUME_MODE" = "auto" ]; then
    tracker=$OUTPUT_DIR/checkpoints/latest_checkpointed_iteration.txt
    [ -f "$tracker" ] && RESUME_STEP=$(cat "$tracker") || RESUME_STEP=0
fi

cat <<BANNER
=========================== RouteWeaver =======================================
  train parquet    : $TRAIN_FILES
  val              : $VAL_FILES
                     test_freq=$TEST_FREQ  val_only=$VAL_ONLY  val_before_train=$VAL_BEFORE_TRAIN
  router POLICY    : $MODEL_PATH   (local weights, FSDP2 + vLLM async)
  frozen workers   : $WORKER_IDS  @ temperature $WORKER_TEMPERATURE
  channel config   : $CHANNEL_CONFIG
  agent loop       : $AGENT_LOOP  ($AGENT_LOOP_CONFIG)
  reward manager   : TaskRewardManager ($REPO/task_reward_manager.py)  reward {-1,0,+1}
  adv estimator    : $ADV_ESTIMATOR  + dispatch gate (infra_failed)
  grouping         : uid = (sample_id, forced_mode), K=$ROLLOUT_N; zero-variance -> 0
  batch/mini/micro : $TRAIN_BATCH_SIZE / $PPO_MINI_BATCH_SIZE (x n) / $MICRO_BATCH_PER_GPU
  router sampling  : temp $ROUTER_TEMPERATURE top_p $ROUTER_TOP_P top_k $ROUTER_TOP_K (overrides generation_config.json)
  prompt/response  : $MAX_PROMPT_LENGTH / $MAX_RESPONSE_LENGTH   obs/turn $MAX_OBS_LENGTH
  memory           : vllm util $GPU_MEM_UTIL, fused kernels $USE_FUSED_KERNELS, remove_padding True
  host RAM         : ${FREE_RAM_GB}GB free (need $MIN_FREE_RAM_GB), ray object store $((OBJECT_STORE_BYTES/1073741824))GB
  dispatch conc.   : concurrency = $DISPATCH_CONCURRENCY  (router.dispatch_concurrency; x $AGENT_LOOP_WORKERS agent-loop worker)
  data seed/shuffle: $DATA_SEED / $DATA_SHUFFLE  (sequential = quota order)
  total steps      : $TOTAL_TRAINING_STEPS   save steps: $SAVE_STEPS
  cost             : COST_ALPHA=${COST_ALPHA:-0}  C_REF=${C_REF:-<unset>}  manager=${REWARD_MANAGER_NAME:-TaskRewardManager}
  resume mode      : $RESUME_MODE   starting at step $RESUME_STEP
  GPUs             : $CUDA_DEVICES  (n_gpus_per_node=$N_GPUS)
  wandb            : $PROJECT_NAME / $WANDB_GROUP / $EXPERIMENT_NAME
                     run id $(cat "$OUTPUT_DIR/wandb_run_id" 2>/dev/null || echo '<new>') (resume=allow: one run across relaunches)
  local metrics    : ${VERL_FILE_LOGGER_PATH:-<set at launch>}
  checkpoints      : $OUTPUT_DIR/checkpoints
  rollout dump     : $OUTPUT_DIR/rollout_dump
  extra overrides  : $*
===============================================================================
BANNER

# Log the alias -> backend mapping for this run.
if [ "${ROUTEWEAVER_WORKER_ALIAS:-0}" = "1" ]; then
    PYTHONPATH="$REPO/routing:${PYTHONPATH:-}" python3 -c \
        "from worker_alias import mapping_banner; print(mapping_banner())"
    echo "  channel config   : $CHANNEL_CONFIG   (its workers: keys must be the aliases)"
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 -- resolved above, nothing launched."
    exit 0
fi

echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
# Source directories on PYTHONPATH (the repo root is deliberately excluded).
export PYTHONPATH=$SRC_DIRS
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export ROUTEWEAVER_SAVE_STEPS=$SAVE_STEPS
export ROUTEWEAVER_METRICS=1                      # routeweaver/* step metrics hook

# ---- provenance, stamped into the W&B run config on the first step ----------
export ROUTEWEAVER_GIT_COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
export ROUTEWEAVER_GIT_BRANCH=$(git -C "$REPO" branch --show-current 2>/dev/null)
export ROUTEWEAVER_INIT_CKPT="$MODEL_PATH"
export ROUTEWEAVER_INIT_GLOBAL_STEP="${INIT_GLOBAL_STEP:-0}"
export ROUTEWEAVER_TRAIN_PARQUET="$TRAIN_FILES"
export ROUTEWEAVER_TRAIN_PARQUET_SHA=$("$PY" -c "
import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])
" "$TRAIN_FILES" 2>/dev/null)
export ROUTEWEAVER_PROBE_SET="${PROBE_SET:-}"
export ROUTEWEAVER_PROBE_SHA="${PROBE_SHA:-}"
export ROUTEWEAVER_SEED="${SEED:-$DATA_SEED}"   # provenance only
export ROUTEWEAVER_REWARD_DEF="+1 correct | 0 legal-but-wrong | -1 policy grammar/protocol invalid (terminal) | infra failure retried then excluded from the advantage group, never -1"
export ROUTEWEAVER_ROUTING_STATE="$ROUTING_STATE"
# ---- minimum-support auxiliary loss (Eq. 12). Inert unless ROUTEWEAVER_SUPPORT=1.
export ROUTEWEAVER_SUPPORT="${ROUTEWEAVER_SUPPORT:-0}"
export SUPPORT_LAMBDA0="${SUPPORT_LAMBDA0:-0.002}"
export SUPPORT_FLOOR="${SUPPORT_FLOOR:-0.05}"
export SUPPORT_CHUNK="${SUPPORT_CHUNK:-12}"
export SUPPORT_MAX_LEN="${SUPPORT_MAX_LEN:-2048}"
export ROUTEWEAVER_TOKENIZER="${ROUTEWEAVER_TOKENIZER:-$MODEL_PATH}"
export FORCED_START_BATCH FORCED_ANNEAL_STEPS
export ROUTEWEAVER_RESUME_STEP="$RESUME_STEP"   # support hook: step offset on a resumed run
# ---- hierarchical advantage on free prompts (comet_grpo.py). Inert unless ROUTEWEAVER_COMET=1.
export ROUTEWEAVER_COMET="${ROUTEWEAVER_COMET:-0}"
# global = one token-mean over the whole response (verl's own default)
# segment = mean over the mode span + lambda_inner * mean over the inner span
export ROUTEWEAVER_LOSS_REDUCTION="${ROUTEWEAVER_LOSS_REDUCTION:-global}"
export ROUTEWEAVER_HOOK_METRICS_DIR="${ROUTEWEAVER_HOOK_METRICS_DIR:-$OUTPUT_DIR/hook_metrics}"
if [ "$ROUTEWEAVER_COMET" = "1" ]; then
    case "$ADV_ESTIMATOR" in
      comet_grpo|comet_gdpo) ;;
      *) echo "FATAL: ROUTEWEAVER_COMET=1 needs ADV_ESTIMATOR=comet_grpo or comet_gdpo" >&2; exit 1;;
    esac
    echo "  hierarchical adv : ON  (free prompts: A_mode on the mode span, A_inner after it; forced prompts: stock GRPO)"
    echo "  loss reduction   : $ROUTEWEAVER_LOSS_REDUCTION  (segment = sum(mode)/N_mode + sum(inner)/N_inner, both counts mini-batch-global)"
fi
export ROUTEWEAVER_POLICY_VIOLATION_RECOVERY="$POLICY_VIOLATION_RECOVERY"
export ROUTEWEAVER_ROLLOUT_N="$ROLLOUT_N"
export MODE_SCORE_SPAN="${MODE_SCORE_SPAN:-full}"   # content = score only the mode content tokens
export VERL_FILE_LOGGER_PATH
# One W&B run id across relaunches.
RUN_ID_FILE=$OUTPUT_DIR/wandb_run_id
if [ ! -f "$RUN_ID_FILE" ]; then
    echo "${WANDB_RUN_ID:-$EXPERIMENT_NAME}" > "$RUN_ID_FILE"
fi
export WANDB_RUN_ID=$(cat "$RUN_ID_FILE")
export WANDB_RESUME=allow
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_RUN_GROUP=$WANDB_GROUP
export WANDB_NAME=$EXPERIMENT_NAME
export WANDB_DIR=${WANDB_DIR:-$OUTPUT_DIR}
unset RAY_ADDRESS

cd "$V1"

set +e
"$PY" -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.shuffle="$DATA_SHUFFLE" \
  data.seed="$DATA_SEED" \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH_PER_GPU" \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  `# The reference model stays on the GPU.` \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_PER_GPU" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.temperature="$ROUTER_TEMPERATURE" \
  actor_rollout_ref.rollout.top_p="$ROUTER_TOP_P" \
  actor_rollout_ref.rollout.top_k="$ROUTER_TOP_K" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-False}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-0}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}" \
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE:-False}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH_PER_GPU" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_LOOP_WORKERS" \
  actor_rollout_ref.rollout.agent.default_agent_loop="$AGENT_LOOP" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD:-True}" \
  algorithm.use_kl_in_reward=False \
  `# One reward worker (host RAM bound).` \
  reward.num_workers=1 \
  reward.reward_manager.source=importlib \
  reward.reward_manager.name="${REWARD_MANAGER_NAME:-TaskRewardManager}" \
  reward.reward_manager.module.path="${REWARD_MANAGER_PATH:-$REPO/rewards/task_reward_manager.py}" \
  +router.max_obs_length="$MAX_OBS_LENGTH" \
  +router.channel_config_path="$CHANNEL_CONFIG" \
  +router.dispatch_concurrency="$DISPATCH_CONCURRENCY" \
  +router.selector_max_tokens="$SELECTOR_MAX_TOKENS" \
  +router.worker_ids="'$WORKER_IDS'" \
  +router.worker_temperature="$WORKER_TEMPERATURE" \
  +router.worker_max_tokens="$WORKER_MAX_TOKENS" \
  +router.max_workers_v2="$MAX_WORKERS_V2" \
  +router.policy_violation_recovery="$POLICY_VIOLATION_RECOVERY" \
  +router.routing_state="$ROUTING_STATE" \
  +router.forced_start_batch="$FORCED_START_BATCH" \
  +router.forced_anneal_steps="$FORCED_ANNEAL_STEPS" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq=1 \
  trainer.test_freq="$TEST_FREQ" \
  trainer.max_actor_ckpt_to_keep="$MAX_CKPT_KEEP" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.val_only="$VAL_ONLY" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.resume_from_path="$RESUME_FROM_PATH" \
  trainer.critic_warmup=0 \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.default_local_dir="$OUTPUT_DIR/checkpoints" \
  trainer.rollout_data_dir="$OUTPUT_DIR/rollout_dump" \
  trainer.validation_data_dir="$OUTPUT_DIR/validation_dump" \
  `# Cap Ray's object store (host RAM).` \
  +ray_kwargs.ray_init.object_store_memory=$OBJECT_STORE_BYTES \
  +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE='"1"' \
  +ray_kwargs.ray_init.runtime_env.env_vars.ROUTEWEAVER_SAVE_STEPS="'$SAVE_STEPS'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.ROUTEWEAVER_METRICS='"1"' \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH="$VERL_FILE_LOGGER_PATH" \
  +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RUN_GROUP="$WANDB_GROUP" \
  "$@"
TRAIN_RC=$?
set -e
echo "=== launcher exit: rc=$TRAIN_RC experiment=$EXPERIMENT_NAME steps=$TOTAL_TRAINING_STEPS ==="
exit "$TRAIN_RC"
