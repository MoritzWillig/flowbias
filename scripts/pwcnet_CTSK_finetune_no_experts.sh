#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/dataA/experiments
#EXPERIMENTS_HOME=/data/vimb01/experiments

# model and checkpoint
MODEL=PWCNetWOX1Connection
EVAL_LOSS=MultiScaleAdaptiveEPE_PWC
#CHECKPOINT=/visinf/home/vimb01/projects/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241
CHECKPOINT=/data/dataB/models/expert_base_wox1_PWCNetWOX1Connection-20200309-153241
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/pwc_noExperts_finetune_CTSK_$MODEL-$TIME"
SAVE_EVERY=1

# set cuda GPU ids
export CUDA_VISIBLE_DEVICES=0

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[20, 30, 40]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-5 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--save_every_nth_checkpoint=$SAVE_EVERY \
--total_epochs=50 \
--training_sampler=CTSKTrainDatasetBatchSampler \
--training_iters_per_epoch=10000 \
--training_augmentation=RandomAffineFlowAdaptive \
--training_dataset=CTSKTrain \
--training_dataset_photometric_augmentations=True \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=CTSKValid  \
--validation_dataset_photometric_augmentations=False \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
