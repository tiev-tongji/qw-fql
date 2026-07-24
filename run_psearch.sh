#!/bin/bash
# dfq 参数搜索 —— 8卡并行，每个任务 seed=8,9 独占一张卡
# 用法:
#   bash run_psearch.sh
#   bash run_psearch.sh --env=antmaze-large-navigate-singletask-v0 --alphas="3 10 30" --temps="0.01 0.1"

set -u

# ---------- CLI 参数解析 ----------
ENV_ARG=""
ALPHA_ARG=""
TEMP_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env=*)    ENV_ARG="${1#*=}"; shift ;;
        --alphas=*) ALPHA_ARG="${1#*=}"; shift ;;
        --temps=*)  TEMP_ARG="${1#*=}"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_GPUS=8
LOG_DIR="logs/grid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ---------- 公共 agent 参数 ----------
AGENT_FLAGS="--agent=agents/dfq.py"

# ---------- 环境及其专属 discount ----------
ALL_ENVS=(
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

if [ -n "$ALPHA_ARG" ]; then
    read -ra ALPHAS <<< "$ALPHA_ARG"
else
    ALPHAS=(3 10 30 100 300 1000)
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
    for alpha in "${ALPHAS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                TASKS+=("${env}|${alpha}|${temp}|${seed}")
        done
    done
done
done

total=${#TASKS[@]}
echo "Search grid: ${#ENVS[@]} envs × ${#ALPHAS[@]} alphas × ${#TEMPS[@]} temps × ${#SEEDS[@]} seeds = ${total} tasks"
echo "$NUM_GPUS GPUs in parallel"
echo "Logs: ${LOG_DIR}"
echo ""

# ---------- 并行执行 ----------
i=0
gpu=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r env alpha temp seed <<< "$task"
    i=$((i + 1))
    tag="${env%%-*}_a${alpha}_t${temp}"
    echo "[$i/$total] GPU$gpu  env=$env  alpha=$alpha  temp=$temp  seed=$seed"

    uv run main.py \
        --env_name="$env" \
        ${AGENT_FLAGS} \
        ${DISCOUNT[$env]:-} \
        --agent.alpha="$alpha" \
        --agent.va_temperature="$temp" \
        --seeds="${seed}" \
        --gpu_ids="$gpu" \
        > "${LOG_DIR}/${tag}.log" 2>&1 &

    gpu=$(( (gpu + 1) % NUM_GPUS ))

    # 已有 NUM_GPUS 个在跑时，等任意一个结束再放新任务
    while [ "$(jobs -r | wc -l)" -ge "$NUM_GPUS" ]; do
        wait -n
    done
done
wait

echo "=========================================="
echo "All experiments completed!"
echo "Logs in: ${LOG_DIR}"
echo "=========================================="