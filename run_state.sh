#!/bin/bash

COMMON_FLAGS="--agent=agents/d3fql.py --agent.num_ensembles=5 --agent.bc_candidates=10 --agent.critic_lcb_kappa=0.5 --agent.lcb_kappa=0.5 --agent.beta=1.0 --gpu_ids=0,1,2,3,4,5,6,7 --seeds=0,1,2,3,4,5,6,7"
TOTAL=50
IDX=0

# ============================================================
# [1-5] antmaze-large-navigate (alpha=10, beta=10, q_agg=min)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] antmaze-large-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=antmaze-large-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.alpha=3 --agent.qw_temperature=0.001
done

# ============================================================
# [6-10] antmaze-giant-navigate (discount=0.995, alpha=10, beta=10, q_agg=min)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] antmaze-giant-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=antmaze-giant-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=10 --agent.qw_temperature=0.01
done

# ============================================================
# [11-15] humanoidmaze-medium-navigate (discount=0.995, alpha=30, beta=30)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] humanoidmaze-medium-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=humanoidmaze-medium-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=30 --agent.qw_temperature=0.01
done

# ============================================================
# [16-20] humanoidmaze-large-navigate (discount=0.995, alpha=30, beta=30)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] humanoidmaze-large-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=humanoidmaze-large-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=30 --agent.qw_temperature=0.001
done

# ============================================================
# [21-25] antsoccer-arena-navigate (discount=0.995, alpha=10, beta=10)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] antsoccer-arena-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=antsoccer-arena-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=10 --agent.qw_temperature=0.001
done

# ============================================================
# [26-30] cube-single-play (alpha=300, beta=300)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] cube-single-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=cube-single-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.alpha=300 --agent.qw_temperature=0.01
done

# ============================================================
# [31-35] cube-double-play (alpha=300, beta=300)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] cube-double-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=cube-double-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.alpha=100 --agent.qw_temperature=0.01
done

# ============================================================
# [36-40] scene-play (alpha=300, beta=300)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] scene-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=scene-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.alpha=300 --agent.qw_temperature=0.001
done

# ============================================================
# [41-45] puzzle-3x3-play (alpha=1000, beta=1000)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] puzzle-3x3-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=puzzle-3x3-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.alpha=30 --agent.qw_temperature=0.001
done

# ============================================================
# [46-50] puzzle-4x4-play (alpha=1000, beta=1000)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] puzzle-4x4-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=puzzle-4x4-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.alpha=1000 --agent.qw_temperature=0.001
done

echo "=========================================="
echo "All ${TOTAL} experiments completed!"
echo "=========================================="
