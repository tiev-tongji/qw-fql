#!/bin/bash

COMMON_FLAGS="--agent=agents/d3fql.py --agent.num_ensembles=5 --agent.bc_candidates=5 --agent.critic_lcb_kappa=1.0 --agent.lcb_kappa=1.0 --agent.qw_temperature=1.0 --agent.beta=0.9 --gpu_ids=0,1,2,3,4,5,6,7 --seeds=0,1,2,3,4,5,6,7"

uv run main.py --env_name=antmaze-large-navigate-singletask-v0 ${COMMON_FLAGS} --agent.alpha=10
uv run main.py --env_name=antmaze-giant-navigate-singletask-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=10
uv run main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=30
uv run main.py --env_name=humanoidmaze-large-navigate-singletask-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=30
uv run main.py --env_name=antsoccer-arena-navigate-singletask-v0 ${COMMON_FLAGS} --agent.discount=0.995 --agent.alpha=10
uv run main.py --env_name=cube-single-play-singletask-v0 ${COMMON_FLAGS} --agent.alpha=300
uv run main.py --env_name=cube-double-play-singletask-v0 ${COMMON_FLAGS} --agent.alpha=300
uv run main.py --env_name=scene-play-singletask-v0 ${COMMON_FLAGS} --agent.alpha=300
uv run main.py --env_name=puzzle-3x3-play-singletask-v0 ${COMMON_FLAGS} --agent.alpha=1000
uv run main.py --env_name=puzzle-4x4-play-singletask-v0 ${COMMON_FLAGS} --agent.alpha=1000

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
