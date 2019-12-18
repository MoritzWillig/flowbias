#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/vimb01/experiments

# datasets
FLYINGTHINGS_HOME=/data/vimb01/FlyingThings3D_subset/

# model and checkpoint
MODEL=FlowNet1S
EVAL_LOSS=MultiScaleEPE_FlowNet
CHECKPOINT=None
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-onThings-$TIME"

# set cuda GPU ids
export CUDA_VISIBLE_DEVICES=5

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[110, 147, 183]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--total_epochs=220 \
--training_augmentation=RandomAffineFlow \
--training_dataset=FlyingThings3dCleanTrain \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$FLYINGTHINGS_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=FlyingThings3dCleanValid  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$FLYINGTHINGS_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS