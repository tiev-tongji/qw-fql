#!/bin/bash
# dfq 参数搜索 —— 8卡并行，每卡同时跑 PROCS_PER_GPU 个进程，seed=8,9
# 用法:
#   bash run_psearch.sh
#   bash run_psearch.sh --env=antmaze-large-navigate-singletask-v0 --DQ_RATIO="3 10 30" --temps="0.01 0.1"

set -u

# ---------- CLI 参数解析 ----------
ENV_ARG=""
DQRATIO_ARG=""
TEMP_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env=*)    ENV_ARG="${1#*=}"; shift ;;
        --DQ_RATIO=*) DQRATIO_ARG="${1#*=}"; shift ;;
        --temps=*)  TEMP_ARG="${1#*=}"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_GPUS=8
PROCS_PER_GPU=2
MAX_PROCS=$((NUM_GPUS * PROCS_PER_GPU))
# 每卡多进程时限制单进程显存预分配比例，避免第二个进程 OOM
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
GRID_TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/grid_${GRID_TS}"
mkdir -p "$LOG_DIR"

# ---------- 公共 agent 参数 ----------
AGENT_FLAGS="--agent=agents/dfq.py --offline_steps=1000000"

# ---------- 环境及其专属 discount ----------
ALL_ENVS=(
    # antmaze-umaze-v2
    # antmaze-umaze-diverse-v2
    # antmaze-medium-play-v2
    # antmaze-medium-diverse-v2
    # antmaze-large-play-v2
    # antmaze-large-diverse-v2
    antmaze-large-navigate-singletask-v0
    antmaze-giant-navigate-singletask-v0
    humanoidmaze-medium-navigate-singletask-v0
    humanoidmaze-large-navigate-singletask-v0
    antsoccer-arena-navigate-singletask-v0
    cube-single-play-singletask-v0
    cube-double-play-singletask-v0
    scene-play-singletask-v0
    puzzle-3x3-play-singletask-v0
    puzzle-4x4-play-singletask-v0
)
declare -A DISCOUNT=(
    [antmaze-giant-navigate-singletask-v0]="--agent.discount=0.995"
    [humanoidmaze-medium-navigate-singletask-v0]="--agent.discount=0.995"
    [humanoidmaze-large-navigate-singletask-v0]="--agent.discount=0.995"
    [antsoccer-arena-navigate-singletask-v0]="--agent.discount=0.995"
)

# ---------- 搜索网格（CLI 优先，否则用默认值） ----------
if [ -n "$ENV_ARG" ]; then
    read -ra ENVS <<< "$ENV_ARG"
else
    ENVS=("${ALL_ENVS[@]}")
fi

if [ -n "$DQRATIO_ARG" ]; then
    read -ra DQ_RATIO <<< "$DQRATIO_ARG"
else
    DQ_RATIO=(0.1 0.3 1.0 3.0 10.0)
fi

if [ -n "$TEMP_ARG" ]; then
    read -ra TEMPS <<< "$TEMP_ARG"
else
    TEMPS=(0.01 0.1 0.5 1.0 2.0)
fi

SEEDS=(8 9)

# ---------- 构建任务队列 ----------
TASKS=()
for env in "${ENVS[@]}"; do
    for target_distill_q_grad_ratio in "${DQ_RATIO[@]}"; do
        for temp in "${TEMPS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                TASKS+=("${env}|${target_distill_q_grad_ratio}|${temp}|${seed}")
        done
    done
done
done

total=${#TASKS[@]}
echo "Search grid: ${#ENVS[@]} envs × ${#DQ_RATIO[@]} DQ_RATIO × ${#TEMPS[@]} temps × ${#SEEDS[@]} seeds = ${total} tasks"
echo "$MAX_PROCS processes in parallel ($NUM_GPUS GPUs × $PROCS_PER_GPU/GPU, mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION)"
echo "Logs: ${LOG_DIR}"
echo ""

# ---------- 并行执行 ----------
i=0
gpu=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r env target_distill_q_grad_ratio temp seed <<< "$task"
    i=$((i + 1))
    tag="${env}_r${target_distill_q_grad_ratio}_t${temp}_s${seed}"
    # 为每个 (env, ratio) 组合指定唯一 time_str，避免同秒启动时目录冲突
    # （main.py 的 exp_name 只含 alpha/va_temperature，不含 ratio）
    time_str="r${target_distill_q_grad_ratio}_${GRID_TS}"
    echo "[$i/$total] GPU$gpu  env=$env  target_distill_q_grad_ratio=$target_distill_q_grad_ratio  temp=$temp  seed=$seed"

    # launcher 进程绑定到对应 GPU 并禁用显存预分配（仅训练子进程预分配），
    # 否则多个 launcher 会同时抢占 GPU 0 导致 OOM
    CUDA_VISIBLE_DEVICES="$gpu" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION" \
    uv run main.py \
        --env_name="$env" \
        ${AGENT_FLAGS} \
        ${DISCOUNT[$env]:-} \
        --agent.target_distill_q_grad_ratio="$target_distill_q_grad_ratio" \
        --agent.va_temperature="$temp" \
        --seeds="${seed}" \
        --gpu_ids="$gpu" \
        --time_str="$time_str" \
        > "${LOG_DIR}/${tag}.log" 2>&1 &

    gpu=$(( (gpu + 1) % NUM_GPUS ))

    # 已有 MAX_PROCS 个在跑时，等任意一个结束再放新任务
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PROCS" ]; do
        wait -n
    done
done
wait

echo "=========================================="
echo "All experiments completed!"
echo "Logs in: ${LOG_DIR}"
echo "=========================================="