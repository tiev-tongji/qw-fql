#!/bin/bash

COMMON_FLAGS="--agent=agents/qw_fql.py --agent.bc_candidates=10 --agent.qw_top_k=3 --gpu_ids=0,1,2,3,4,5,6,7 --seeds=0,1,2,3,4,5,6,7"
TOTAL=3
IDX=0

# [1] visual-cube-single-play
IDX=$((IDX + 1))
echo "=========================================="
echo "[${IDX}/${TOTAL}] visual-cube-single-play-singletask-task1-v0"
echo "=========================================="
uv run main.py --env_name=visual-cube-single-play-singletask-task1-v0 ${COMMON_FLAGS} --offline_steps=500000 --agent.alpha=300 --agent.beta=0.5 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3

# [2] visual-cube-double-play
IDX=$((IDX + 1))
echo "=========================================="
echo "[${IDX}/${TOTAL}] visual-cube-double-play-singletask-task1-v0"
echo "=========================================="
uv run main.py --env_name=visual-cube-double-play-singletask-task1-v0 ${COMMON_FLAGS} --offline_steps=500000 --agent.alpha=100 --agent.beta=0.5 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3

# [3] visual-scene-play
IDX=$((IDX + 1))
echo "=========================================="
echo "[${IDX}/${TOTAL}] visual-scene-play-singletask-task1-v0"
echo "=========================================="
uv run main.py --env_name=visual-scene-play-singletask-task1-v0 ${COMMON_FLAGS} --offline_steps=500000 --agent.alpha=100 --agent.beta=0.5 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3

# [4] visual-puzzle-3x3-play
IDX=$((IDX + 1))
echo "=========================================="
echo "[${IDX}/${TOTAL}] visual-puzzle-3x3-play-singletask-task1-v0"
echo "=========================================="
uv run main.py --env_name=visual-puzzle-3x3-play-singletask-task1-v0 ${COMMON_FLAGS} --offline_steps=500000 --agent.alpha=300 --agent.beta=0.5 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3

# [5] visual-puzzle-4x4-play
IDX=$((IDX + 1))
echo "=========================================="
echo "[${IDX}/${TOTAL}] visual-puzzle-4x4-play-singletask-task1-v0"
echo "=========================================="
uv run main.py --env_name=visual-puzzle-4x4-play-singletask-task1-v0 ${COMMON_FLAGS} --offline_steps=500000 --agent.alpha=300 --agent.beta=0.5 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3


echo "=========================================="
echo "All ${TOTAL} pixel experiments completed!"
echo "=========================================="
