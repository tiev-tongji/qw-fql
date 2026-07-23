#!/bin/bash
# grid_search.sh —— 8卡并行，每个任务 seed=0 独占一张卡

set -u
NUM_GPUS=8
LOG_DIR="logs/grid_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ---------- 公共 agent 参数（不含 alpha / qw_temperature / seeds / gpu_ids） ----------
AGENT_FLAGS="--agent=agents/d3fql.py --agent.num_ensembles=5 --agent.bc_candidates=10 --agent.critic_lcb_kappa=0.5 --agent.lcb_kappa=0.5 --agent.beta=1.0"

# ---------- 环境及其专属 discount ----------
ENVS=(
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
)   # 其余环境用默认 discount，不额外传

# ---------- 搜索网格 ----------
ALPHAS=(3 10 30 100 300)
TEMPS=(0.001 0.01 0.1)

# ---------- 构建任务队列 ----------
TASKS=()
for env in "${ENVS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for temp in "${TEMPS[@]}"; do
            TASKS+=("${env}|${alpha}|${temp}")
        done
    done
done

total=${#TASKS[@]}
echo "Total tasks: $total, $NUM_GPUS GPUs in parallel"

# ---------- 并行执行 ----------
i=0
gpu=0
for task in "${TASKS[@]}"; do
    IFS='|' read -r env alpha temp <<< "$task"
    i=$((i + 1))
    tag="${env%%-*}_a${alpha}_t${temp}"
    echo "[$i/$total] GPU$gpu  env=$env  alpha=$alpha  temp=$temp"

    uv run main.py \
        --env_name="$env" \
        ${AGENT_FLAGS} \
        ${DISCOUNT[$env]:-} \
        --agent.alpha="$alpha" \
        --agent.qw_temperature="$temp" \
        --seeds=0 \
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