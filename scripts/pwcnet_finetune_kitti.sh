#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/dataA/experiments

# datasets
#KITTI_HOME=/data/vimb01/KITTI_scene_flow/
KITTI_HOME=kitti

# model and checkpoint
MODEL=PWCNet
EVAL_LOSS=MultiScaleSparseEPE_PWC
#CHECKPOINT=/visinf/home/vimb01/projects/models/A_PWCNet-onChairs-20191121-171532
CHECKPOINT=/data/dataB/models/I_PWCNet-things_20191209-131019
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-I_fine_kitti-$TIME"
EVAL_EVERY=25

# set cuda GPU ids
export CUDA_VISIBLE_DEVICES=0

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--checkpoint_mode=resume_from_best \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[10000, 15000, 20000]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-5 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--eval_every_nth=$EVAL_EVERY \
--total_epochs=25000 \
--training_augmentation=RandomAffineFlowOccKITTI \
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
