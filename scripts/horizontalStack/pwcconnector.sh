#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/vimb01/experiments/pwcconnector_A_B/

# interface datasets
CONNECTOR_HOME_A=/data/vimb01/connectorA/
CONNECTOR_HOME_B=/data/vimb01/connectorB/

# model and checkpoint
MODEL=PWCConnector1
TRAIN_LOSS=L1ConnectorLoss
EVAL_LOSS=L1ConnectorLoss
CHECKPOINT=None
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-$TIME"

# training configuration
python ../../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[200, 500, 700]" \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--total_epochs=1000 \
--training_dataset=PWCInterfaceDatasetTrain \
--training_dataset_rootA=$CONNECTOR_HOME_A \
--training_dataset_rootB=$CONNECTOR_HOME_B \
--training_key=total_loss \
--training_loss=$TRAIN_LOSS \
--validation_dataset=PWCInterfaceDatasetValid  \
--validation_dataset_rootA=$CONNECTOR_HOME_A \
--validation_dataset_rootB=$CONNECTOR_HOME_B \
--validation_key=total_loss \
--validation_loss=$EVAL_LOSS
