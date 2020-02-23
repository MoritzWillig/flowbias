#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/vimb01/experiments
#EXPERIMENTS_HOME=/data/dataA/experiments

# datasets
#KITTI_HOME=/data02/vimb01/KITTI_scene_flow/
KITTI_HOME=kitti

# model and checkpoint
MODEL=PWCNet
EVAL_LOSS=MultiScaleSparseEPE_PWC
CHECKPOINT=None
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-kitti-$TIME"
SAVE_EVERY=150
EVAL_EVERY=25

# set cuda GPU ids
export CUDA_VISIBLE_DEVICES=1

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[15000, 20000, 25000]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--save_every_nth_checkpoint=$SAVE_EVERY \
--eval_every_nth=$EVAL_EVERY \
--total_epochs=30000 \
--training_augmentation=RandomAffineFlow \
--training_dataset=KittiComb2015Train \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$KITTI_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=KittiComb2015Val  \
--validation_dataset_preprocessing_crop=False \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$KITTI_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
