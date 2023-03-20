#!/usr/bin/bash

SIZE=4
ALGO=ppo
NET_ARCH="64 64 64"
TIMESTEPS=1000000
GAMMA=0.95
EXP_NAME=${ALGO}_${SIZE}_arch=${NET_ARCH// /-}

cmd=(
    python train_sb3.py
    --size $SIZE
    --algo $ALGO
    --net-arch ${NET_ARCH[@]}
    --gamma $GAMMA
    --train-timesteps $TIMESTEPS
    --random-start-pos --random-goal-pos
    --obscure-type neighbor
)

# Uncomment this line to train an agent with hidden holes
# cmd+=(--ratio-hide 0.5)
# EXP_NAME+=_hide=0.5

# GUIDE AGENT OPTIONS

# Guide schedule: always, random, time, hole, hidden_hole
# GUIDE_TYPE=vi
# GUIDE_SCHEDULE=always
# cmd+=(--guide-type $GUIDE_TYPE)
# cmd+=(--guide-schedule $GUIDE_SCHEDULE)
# EXP_NAME+=_guide=${GUIDE_TYPE}_${GUIDE_SCHEDULE}

# GUIDE_TYPE=ppo
# GUIDE_SCHEDULE=always
# GUIDE_CKPT=sb3_ckpt/ppo_4_arch=64-64-64
# cmd+=(--guide-type $GUIDE_TYPE)
# cmd+=(--guide-schedule $GUIDE_SCHEDULE)
# cmd+=(--guide-ckpt $GUIDE_CKPT)
# EXP_NAME+=_guide=${GUIDE_TYPE}_${GUIDE_SCHEDULE}

# Uncomment this line to resume training from a checkpoint
# cmd+=(--init-ckpt sb3_ckpt/$EXP_NAME)

# Uncomment this line to resume training from a checkpoint
# NOTE: Make a modification to the name if you run different initializations
# INIT_EXP=ppo_4_arch=64-64-64
# cmd+=(--init-ckpt sb3_ckpt/$INIT_EXP)
# EXP_NAME+=_init

# Uncomment this line to resume training from a checkpoint
# NOTE: Make a modification to the name if you run different initializations
# INIT_EXP=ppo_4_arch=64-64-64_hide=0.5
# cmd+=(--init-ckpt sb3_ckpt/$INIT_EXP)
# EXP_NAME+=_init=hide

cmd+=(--exp-name $EXP_NAME)

# Optionally print the command in copy-pasteable format
echo "Executing: "
printf "%q " "${cmd[@]}"
echo

# Execute the command:
"${cmd[@]}"
