# Settings shared by every run, sourced by scripts/train.sh and
# scripts/eval.sh. Per-run settings -- advantage estimator, loss reduction,
# support term, schedule, resume point -- live in those two scripts.
# Everything here is the paper's default; see Appendix F.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=$REPO/data
# Built datasets live outside the repo (nothing in datasets/ is tracked).
DATASET_ROOT=${ROUTEWEAVER_DATASETS:-$REPO/datasets}
OUTPUTS=${OUTPUTS:-${ROUTEWEAVER_OUTPUTS:-$REPO/outputs}}
PY=${PY:-python}
export PY DATASET_ROOT

# ---- router, data, batch --------------------------------------------------
export MODEL_PATH=${MODEL_PATH:-${ROUTER_MODEL:-$HOME/models/Qwen3-4B-Instruct-2507}}
export TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=32 ROLLOUT_N=8
export NORM_ADV_BY_STD=True

# ---- environment: six anonymous workers (Appendix D) ----------------------
export ROUTEWEAVER_WORKER_ALIAS=1
export WORKER_IDS=worker_1,worker_2,worker_3,worker_4,worker_5,worker_6
export CHANNEL_CONFIG=$REPO/configs/route_channels.yaml
export MAX_WORKERS_V2=8            # agentic: at most eight worker calls
export ROUTING_STATE=1
export ROUTEWEAVER_CODE_SCORER=union   # code: worker code blocks are candidates too (Appendix C)
export WORKER_EXPLORE=0 WORKER_ORDER_SHUFFLE=0
export DISPATCH_CONCURRENCY=${DISPATCH_CONCURRENCY:-40}

# ---- agent loop and progressive exploration (Eq. 11) ----------------------
export AGENT_LOOP=routeweaver
export AGENT_LOOP_CONFIG=$REPO/configs/agent_loop.yaml
export FORCED_START_BATCH=25 FORCED_ANNEAL_STEPS=50   # b0 = 25, b1 = 75
export MODE_SCORE_SPAN=content                 # support scores: mode content tokens only
export SUPPORT_LAMBDA0=0.002                   # lambda_b = 0.002 * rho_b, delta = 0.05

export ROUTEWEAVER_METRICS=1

# ---- cost model (Appendix E) ------------------------------------------------
# C_ref is frozen configuration, not a live measurement: the P95 trajectory
# cost over the calibration rollouts. scripts/train.sh calibration regenerates
# it; the efficiency reward reads it from this file.
C_REF_FILE=$REPO/configs/cost_ref.json

TRAIN_PARQUET=$DATASET_ROOT/parquet/routeweaver_train.parquet
COLD_PARQUET=$DATASET_ROOT/parquet/routeweaver_cold_pool.parquet
