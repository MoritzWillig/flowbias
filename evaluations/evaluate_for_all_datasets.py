import json
import sys
import os
from argparse import Namespace
from datetime import datetime
import torch
import torch.utils.data as data
import numpy as np

from flowbias.datasets.flyingchairs import FlyingChairsTrain, FlyingChairsValid, FlyingChairsFull
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanTrain, FlyingThings3dCleanValid
from flowbias.datasets.kitti_combined import KittiComb2015Train, KittiComb2015Val, KittiComb2015Test
from flowbias.datasets.sintel import SintelTrainingCleanTrain, SintelTrainingCleanValid, SintelTrainingCleanFull, \
    SintelTrainingFinalTrain, SintelTrainingFinalValid, SintelTrainingFinalFull
from flowbias.datasets.middlebury import MiddleburyTrainValid
from flowbias.models import PWCNet, FlowNet1S, PWCNetConv33Fusion, PWCNetX1Zero, PWCNetWOX1Connection, \
    CTSKPWCExpertNet02, CTSKPWCExpertNetAdd01, PWCNetDSEncoder, PWCNetWOX1ConnectionExt, CTSPWCExpertNetAdd01, \
    CTSKPWCExpertNet02WOX1, CTSKPWCExpertNetWOX1Add01, CTSPWCExpertNetWOX1Add01,\
    CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly, CTSKPWCExpertNetWOX1LinAdd01, PWCNetResidualFlow, \
    PWCNetWOX1SecondaryFlow
from flowbias.utils.meta_infrastructure import get_available_datasets, dataset_needs_batch_size_one
from flowbias.utils.model_loading import load_model_parameters, sample_to_torch_batch
from flowbias.losses import MultiScaleEPE_PWC, MultiScaleEPE_FlowNet, MultiScaleSparseEPE_PWC, MultiScaleSparseEPE_FlowNet
from flowbias.utils.statistics import SeriesStatistic
from torch.utils.data.dataloader import DataLoader

"""
Computes the average epe of a model for all datasets.

evaluate_for_all_datasets /path_to/model_checkpoint.ckpt networkName
"""

def contains_nan(results):
    if isinstance(results, list):
        return np.isnan(results).any()
    if isinstance(results, dict):
        return any([contains_nan(val) for val in results.values()])

class DataEnricher(data.Dataset):
    def __init__(self, dataset, additional):
        self._dataset = dataset
        self._additional = additional

    def __getitem__(self, index):
        return {**self._dataset[index], **self._additional}

    def __len__(self):
        return len(self._dataset)


class CTSKDatasetDetector(DataEnricher):
    # this are the dataset indices used by the CTSKTrain CombinedDataset and CTSKTrainDatasetBatchSampler
    _known_datasets = [
        [FlyingChairsTrain, 0],
        [FlyingChairsValid, 0],
        [FlyingChairsFull, 0],
        [FlyingThings3dCleanTrain, 1],
        [FlyingThings3dCleanValid, 1],
        [SintelTrainingCleanTrain, 2],
        [SintelTrainingCleanValid, 2],
        [SintelTrainingCleanFull, 2],
        [SintelTrainingFinalTrain, 2],
        [SintelTrainingFinalValid, 2],
        [SintelTrainingFinalFull, 2],
        [KittiComb2015Train, 3],
        [KittiComb2015Val, 3],
        [KittiComb2015Test, 3],
        [MiddleburyTrainValid, -1]
    ]

    def _detect_dataset_id(self, dataset):
        dataset_id = -1
        for dataset_data in CTSKDatasetDetector._known_datasets:
            if isinstance(dataset, dataset_data[0]):
                #print("detected ", dataset_data[0])
                dataset_id = dataset_data[1]
        #if dataset_id == -1:
        #    raise ValueError("Unknown dataset!")
        return dataset_id

    def __init__(self, dataset, additional):
        super().__init__(dataset, {"dataset": self._detect_dataset_id(dataset), **additional})


if __name__ == '__main__':
    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "preparing ...")

    args = Namespace(**{
        "batch_size": None,
        "cuda": True
    })

    model_classes = {
        "PWCNet": [PWCNet, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        "FlowNet1S": [FlowNet1S, {"default": MultiScaleEPE_FlowNet, "kitti2015Train": MultiScaleSparseEPE_FlowNet, "kitti2015Valid": MultiScaleSparseEPE_FlowNet, "kitti2015Test": MultiScaleSparseEPE_FlowNet, "middleburyTrain":MultiScaleSparseEPE_FlowNet}],
        "PWCNetConv33Fusion": [PWCNetConv33Fusion, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        "PWCNetX1Zero": [PWCNetX1Zero, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        "PWCNetWOX1Connection": [PWCNetWOX1Connection, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        "PWCNetWOX1ConnectionExt": [PWCNetWOX1ConnectionExt, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        # CTKS expert split models
        "CTSKPWCExpertNet02Known": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSKPWCExpertNet02Expert0": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSKPWCExpertNet02Expert1": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSKPWCExpertNet02Expert2": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        "CTSKPWCExpertNet02Expert3": [CTSKPWCExpertNet02, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 3}]],
        # CTKS expert add models
        "CTSKPWCExpertNet01AddKnown": [CTSKPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSKPWCExpertNet01AddNoExpert": [CTSKPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": -1}]],
        "CTSKPWCExpertNet01AddExpert0": [CTSKPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSKPWCExpertNet01AddExpert1": [CTSKPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSKPWCExpertNet01AddExpert2": [CTSKPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        "CTSKPWCExpertNet01AddExpert3": [CTSKPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 3}]],
        # CTKS WOX1 expert split models
        "CTSKPWCExpertNet02WOX1Known": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSKPWCExpertNet02WOX1Expert0": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSKPWCExpertNet02WOX1Expert1": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSKPWCExpertNet02WOX1Expert2": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        "CTSKPWCExpertNet02WOX1Expert3": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 3}]],
        # CTKS WOX1 expert add models
        "CTSKPWCExpertNet01WOX1AddKnown": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSKPWCExpertNet01WOX1AddNoExpert": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": -1}]],
        "CTSKPWCExpertNet01WOX1AddExpert0": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSKPWCExpertNet01WOX1AddExpert1": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSKPWCExpertNet01WOX1AddExpert2": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        "CTSKPWCExpertNet01WOX1AddExpert3": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 3}]],
        # CTKS WOX1 expert linAdd models
        "CTKSPWCExpertLinAddNet01WOX1Known": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTKSPWCExpertLinAddNet01WOX1NoExpert": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": -1}]],
        "CTKSPWCExpertLinAddNet01WOX1Expert0": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTKSPWCExpertLinAddNet01WOX1Expert1": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTKSPWCExpertLinAddNet01WOX1Expert2": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        "CTKSPWCExpertLinAddNet01WOX1Expert3": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 3}]],
        #CTS Expert add Models
        "CTSPWCExpertNet01AddKnown": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSPWCExpertNet01AddNoExpert": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": -1}]],
        "CTSPWCExpertNet01AddExpert0": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSPWCExpertNet01AddExpert1": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSPWCExpertNet01AddExpert2": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        # CTS WOX1 expert add models
        "CTSPWCExpertNet01WOX1AddKnown": [CTSPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSPWCExpertNet01WOX1AddNoExpert": [CTSPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": -1}]],
        "CTSPWCExpertNet01WOX1AddExpert0": [CTSPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSPWCExpertNet01WOX1AddExpert1": [CTSPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSPWCExpertNet01WOX1AddExpert2": [CTSPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        # CTKS WOX1 expert add models encoder only
        "CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyKnown": [CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [CTSKDatasetDetector, {}]],
        "CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyNoExpert": [CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": -1}]],
        "CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert0": [CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 0}]],
        "CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert1": [CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 1}]],
        "CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert2": [CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 2}]],
        "CTSKPWCExpertNetWOX1Add01EncoderExpertsOnlyExpert3": [CTSKPWCExpertNetWOX1Add01EncoderExpertsOnly, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"dataset": 3}]],
        # DS Encoder
        "PWCNetDSEncoder": [PWCNetDSEncoder, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        # pwc secondary flow
        "PWCNetWOX1SecondaryFlow": [PWCNetWOX1SecondaryFlow, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        # Residual Flow
        "PWCNetResidualFlow": [PWCNetResidualFlow, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}],
        # CTS fused expert add models
        "CTSPWCExpertNet01AddExpert00": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSPWCExpertNet01AddExpert01": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSPWCExpertNet01AddExpert02": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSPWCExpertNet01AddExpert10": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSPWCExpertNet01AddExpert11": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSPWCExpertNet01AddExpert12": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSPWCExpertNet01AddExpert20": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSPWCExpertNet01AddExpert21": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSPWCExpertNet01AddExpert22": [CTSPWCExpertNetAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 2, "context_expert_id": 2}]],
        #CTKS WOX1 fused expert split models
        "CTSKPWCExpertNet02WOX1Expert00": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet02WOX1Expert01": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet02WOX1Expert02": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet02WOX1Expert03": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet02WOX1Expert10": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet02WOX1Expert11": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet02WOX1Expert12": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet02WOX1Expert13": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet02WOX1Expert20": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet02WOX1Expert21": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet02WOX1Expert22": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet02WOX1Expert23": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet02WOX1Expert30": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet02WOX1Expert31": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet02WOX1Expert32": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet02WOX1Expert33": [CTSKPWCExpertNet02WOX1, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 3, "context_expert_id": 3}]],
        #CTKS WOX1 fused expert add models
        "CTSKPWCExpertNet01WOX1AddExpert00": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1AddExpert01": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1AddExpert02": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1AddExpert03": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet01WOX1AddExpert10": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1AddExpert11": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1AddExpert12": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1AddExpert13": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet01WOX1AddExpert20": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1AddExpert21": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1AddExpert22": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1AddExpert23": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet01WOX1AddExpert30": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1AddExpert31": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1AddExpert32": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1AddExpert33": [CTSKPWCExpertNetWOX1Add01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 3, "context_expert_id": 3}]],
        #CTKS WOX1 fused expert add models
        "CTSKPWCExpertNet01WOX1LinAddExpert00": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert01": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert02": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert03": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 0, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert10": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert11": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert12": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert13": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 1, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert20": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert21": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert22": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert23": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 2, "decoder_expert_id": 3, "context_expert_id": 3}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert30": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 0, "context_expert_id": 0}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert31": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 1, "context_expert_id": 1}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert32": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 2, "context_expert_id": 2}]],
        "CTSKPWCExpertNet01WOX1LinAddExpert33": [CTSKPWCExpertNetWOX1LinAdd01, {"default": MultiScaleEPE_PWC, "kitti2015Train": MultiScaleSparseEPE_PWC, "kitti2015Valid": MultiScaleSparseEPE_PWC, "kitti2015Test": MultiScaleSparseEPE_PWC, "middleburyTrain":MultiScaleSparseEPE_PWC}, [DataEnricher, {"encoder_expert_id": 3, "decoder_expert_id": 3, "context_expert_id": 3}]],
    }

    assert(len(sys.argv) == 4)

    #model_path = "/data/dataB/models/R_PWCNet-A_fine_sintel-20191218-135407/checkpoint_latest.ckpt"
    model_path = sys.argv[1]
    model_class_name = sys.argv[2]
    result_file_path = sys.argv[3]
    print(model_path, "with", model_class_name)

    with torch.no_grad():
        model_class = model_classes[model_class_name][0]
        model = model_class(args)
        load_model_parameters(model, model_path)
        model.eval().cuda()

        rename = {
            "flyingChairs": "flyingChairsValid",
            "flyingThings": "flyingThingsCleanValid",
            "kitti": "kittiValid",
            "kittiValid": "kitti2015Valid",
            "sintelClean": "sintelCleanValid",
            "sintelFinal": "sintelFinalValid",
        }

        # load existing results
        has_old_names = False
        if os.path.isfile(result_file_path):
            with open(result_file_path, "r") as f:
                existing_results_x = json.loads(f.read())

            # rename old keys and skip non-dataset entries
            existing_results = {}
            for key, value in existing_results_x.items():
                if key in ["model_path", "model_class_name"]:
                    # check if the file contains old model_class_names
                    if key == "model_class_name" and value not in model_class_name:
                        has_old_names = True
                    continue

                if key in rename:
                    existing_results[rename[key]] = value
                    has_old_names = True
                else:
                    existing_results[key] = value
        else:
            # no existing results
            existing_results = {}
        existing_results_datasets = list(existing_results.keys())

        # compute remaining evaluations
        #reevaluate = ["kitti2015Train", "kitti2015Valid"]  # forces datasets to be reevaluated
        #reevaluate = ["middleburyTrain"]
        reevaluate = []
        reevaluate_only = False
        reevaluate_nans = True

        available_dataset_names = get_available_datasets(force_mode="test", select_by_any_tag=["train", "valid"], run_dry=True)
        missing_dataset_names = [
            dataset_name for dataset_name in available_dataset_names
            if (((dataset_name not in existing_results_datasets) or
                 (reevaluate_nans and contains_nan(existing_results[dataset_name]))) and
                (not reevaluate_only)) or (dataset_name in reevaluate)]

        print("available_datasets:", list(available_dataset_names))
        print("existing results:", list(existing_results.keys()))
        print("computing results for:", missing_dataset_names)

        datasets = get_available_datasets(force_mode="test", restrict_to=missing_dataset_names)


        if len(datasets.keys()) == 0:
            if has_old_names:
                print("replacing old dataset or model names")
                results = {"model_path": model_path, "model_class_name": model_class_name}
                for key, value in existing_results.items():
                    results[key] = value
                with open(result_file_path, "w") as f:
                    f.write(json.dumps(results))
            print("no datasets remaining - exiting")
            exit()

        batch_size = 16

        model_config = model_classes[model_class_name]

        demo_available_dataset = next(iter(datasets.values()))
        if len(model_config) > 2:
            # wrap dataset into dataset enricher
            enricherConfig = model_config[2]
            demo_available_dataset = enricherConfig[0](demo_available_dataset, enricherConfig[1])
        demo_sample = sample_to_torch_batch(demo_available_dataset[0])
        demo_loss = model_classes[model_class_name][1]["default"](args).eval().cuda()
        print("!!!!", model_class_name, demo_loss, model(demo_sample).keys(), demo_sample.keys())
        demo_loss_values = demo_loss(model(demo_sample), demo_sample)
        loss_names = list(demo_loss_values.keys())

        results = {"model_path": model_path, "model_class_name": model_class_name}

        for name, dataset in datasets.items():
            print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), name)

            loss_class = model_config[1][name] if name in model_config[1] else model_config[1]["default"]
            loss = loss_class(args).eval().cuda()
            if len(model_config) > 2:
                # wrap dataset into dataset enricher
                enricherConfig = model_config[2]
                dataset = enricherConfig[0](dataset, enricherConfig[1])

            losses = {name: SeriesStatistic() for name in loss_names}
            dataset_size = len(dataset)
            i = 0

            gpuargs = {"num_workers": 4, "pin_memory": False}
            loader = DataLoader(
                dataset,
                batch_size=1 if dataset_needs_batch_size_one(name, force_mode="test") else batch_size,
                shuffle=False,
                drop_last=False,
                **gpuargs)

            #for i in range(len(dataset)):
            for sample in loader:
                input_keys = list(filter(lambda x: "input" in x, sample.keys()))
                target_keys = list(filter(lambda x: "target" in x, sample.keys()))
                tensor_keys = input_keys + target_keys

                for key, value in sample.items():
                    if key in tensor_keys:
                        sample[key] = value.cuda(non_blocking=True)

                loss_values = loss(model(sample), sample)
                for lname, value in loss_values.items():
                    b, _, _, _ = sample["target1"].size()
                    losses[lname].push_value(value.cpu().detach().numpy(), int(b))

                #time.sleep(0.003)

                #i += 1
                #if i+1 % 10 == 0:
                #    sys.stdout.write(f"\r{i}/{dataset_size}")
                #    sys.stdout.flush()
            #sys.stdout.write("\n")
            #sys.stdout.flush()

            results[name] = {}
            for lname, lloss in losses.items():
                statistic = lloss.get_statistics(report_individual_values=True)
                results[name][lname] = statistic
                print(f"{lname}: {statistic['average']}")

    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "saving ...")

    # add existing results
    for key, value in existing_results.items():
        # but keep newer results (in case we reevaluated a dataset)
        if key in results:
            continue
        results[key] = value

    # save
    with open(result_file_path, "w") as f:
        f.write(json.dumps(results))

    print(datetime.now().strftime("[%d-%b-%Y (%H:%M:%S)]"), "done")
