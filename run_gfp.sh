#!/bin/bash

COMMON_FLAGS="--agent=agents/gfp.py --gpu_ids=0,1,2,3,4,5,6,7 --seeds=0,1,2,3,4,5,6,7"
TOTAL=50
IDX=0

# ============================================================
# [1-5] antmaze-giant-navigate (batch=1024, discount=0.995, q_agg=min, guidance_agg=min, alpha=0.1)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] antmaze-giant-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=antmaze-giant-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.q_agg=min --agent.target_agg=actor --agent.guidance_agg=min --agent.alpha=0.1 --agent.eta_temperature=0.1
done

# ============================================================
# [6-10] antmaze-large-navigate (discount=0.995, q_agg=min, alpha=0.3)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] antmaze-large-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=antmaze-large-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.q_agg=min --agent.target_agg=actor --agent.guidance_agg=min --agent.alpha=0.3 --agent.eta_temperature=0.0001
done


# ============================================================
# [11-15] antsoccer-arena-navigate (discount=0.995, guidance_agg=min, alpha=0.1)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] antsoccer-arena-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=antsoccer-arena-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.q_agg=mean --agent.target_agg=actor --agent.guidance_agg=min --agent.alpha=0.1 --agent.eta_temperature=0.01
done

# ============================================================
# [16-20] cube-double-play (alpha=1)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] cube-double-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=cube-double-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.q_agg=mean --agent.target_agg=mean --agent.guidance_agg=actor --agent.alpha=1 --agent.eta_temperature=0.01
done

# ============================================================
# [21-25] cube-single-play (alpha=10)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] cube-single-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=cube-single-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.q_agg=mean --agent.target_agg=mean --agent.guidance_agg=actor --agent.alpha=10 --agent.eta_temperature=0.1
done


# ============================================================
# [26-30] humanoidmaze-large-navigate (batch=1024, discount=0.999, flow_steps=30, alpha=0.3)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] humanoidmaze-large-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=humanoidmaze-large-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.999 --agent.flow_steps=30 --agent.q_agg=mean --agent.target_agg=actor --agent.guidance_agg=min --agent.alpha=0.3 --agent.eta_temperature=0.0001
done

# ============================================================
# [31-35] humanoidmaze-medium-navigate (discount=0.995, guidance_agg=min, alpha=0.3)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] humanoidmaze-medium-navigate-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=humanoidmaze-medium-navigate-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.q_agg=mean --agent.target_agg=mean --agent.guidance_agg=min --agent.alpha=0.3 --agent.eta_temperature=0.001
done

# ============================================================
# [36-40] puzzle-3x3-play (target_agg=actor, alpha=3)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] puzzle-3x3-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=puzzle-3x3-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.q_agg=mean --agent.target_agg=actor --agent.guidance_agg=actor --agent.alpha=3 --agent.eta_temperature=0.001
done

# ============================================================
# [41-45] puzzle-4x4-play (target_agg=actor, alpha=3)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] puzzle-4x4-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=puzzle-4x4-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.q_agg=mean --agent.target_agg=actor --agent.guidance_agg=actor --agent.alpha=3 --agent.eta_temperature=1e-05
done

# ============================================================
# [46-50] scene-play (target_agg=actor, alpha=10)
# ============================================================
for TASK in 1 2 3 4 5; do
    IDX=$((IDX + 1))
    echo "=========================================="
    echo "[${IDX}/${TOTAL}] scene-play-singletask-task${TASK}-v0"
    echo "=========================================="
    uv run main.py --env_name=scene-play-singletask-task${TASK}-v0 ${COMMON_FLAGS} --agent.q_agg=mean --agent.target_agg=actor --agent.guidance_agg=actor --agent.alpha=10 --agent.eta_temperature=0.001
done

echo "=========================================="
echo "All ${TOTAL} experiments completed!"
echo "=========================================="
