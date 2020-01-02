#!/bin/bash

# experiments and datasets meta
EXPERIMENTS_HOME=/data/dataB/models/0_connectors/

# interface datasets
CONNECTOR_HOME_A=/data/dataA/model_interfaces/H_chairs/
CONNECTOR_HOME_B=/data/dataA/model_interfaces/I_chairs/

# model and checkpoint
MODEL=PWCTrainableConvConnector33
TRAIN_LOSS=MSEConnectorLoss
EVAL_LOSS=MSEConnectorLoss
CHECKPOINT=None
SIZE_OF_BATCH=8

# save path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-HI_33_chairs-$TIME"

# training configuration
python ../../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--lr_scheduler=ReduceLROnPlateau \
--lr_scheduler_factor=0.5 \
--lr_scheduler_patience=16 \
--lr_scheduler_cooldown=8 \
--lr_scheduler_threshold=1e-2 \
--lr_scheduler_threshold_mode='rel' \
--model=$MODEL \
--num_workers=4 \
--optimizer=Adam \
--optimizer_lr=1e-3 \
--optimizer_weight_decay=4e-4 \
--save=$SAVE_PATH \
--total_epochs=168 \
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
