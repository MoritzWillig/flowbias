#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/dataA/experiments

# model and checkpoint
MODEL=PWCNetExperts
EVAL_LOSS=MultiScaleAdaptiveEPE_PWC
CHECKPOINT=None
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/expert_base_$MODEL-$TIME"

# set cuda GPU ids
export CUDA_VISIBLE_DEVICES=0

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[110, 147, 183_ADJUST]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--total_epochs=216_ADJUST \
--training_augmentation=RandomAffineFlow \
--training_dataset=CTSKTrain \
--training_dataset_photometric_augmentations=True \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=CTSKValid  \
--validation_dataset_photometric_augmentations=False \
--validation_key=epe \
--validation_loss=$EVAL_LOSS