#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/vimb01/experiments

# datasets
SINTEL_HOME=/data02/vimb01/MPI-Sintel-complete

# model and checkpoint
MODEL=PWCNet
EVAL_LOSS=MultiScaleEPE_PWC
CHECKPOINT=/visinf/home/vimb01/projects/fusedModelsConv33/ia/
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/0C300_$MODEL-fine_sintel-$TIME"

# set cuda GPU ids
export CUDA_VISIBLE_DEVICES=2

# training configuration
python ../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--checkpoint_mode=resume_from_best \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[1504, 2255, 3008]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-5 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--total_epochs=3759 \
--training_augmentation=RandomAffineFlow \
--training_dataset=SintelTrainingCleanTrain \
--training_dataset_photometric_augmentations=True \
--training_dataset_root=$SINTEL_HOME \
--training_key=total_loss \
--training_loss=$EVAL_LOSS \
--validation_dataset=SintelTrainingCleanValid  \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$SINTEL_HOME \
--validation_key=epe \
--validation_loss=$EVAL_LOSS
