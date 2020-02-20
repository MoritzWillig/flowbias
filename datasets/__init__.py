from . import flyingchairs
from . import sintel
from . import flyingThings3D
from . import kitti_combined
from . import sintel
from . import monkaa
from . import pwcInterfaceDataset
from . import subsampledDataset
from . import combined_dataset

## FlyingChairs
FlyingChairsTrain = flyingchairs.FlyingChairsTrain
FlyingChairsValid = flyingchairs.FlyingChairsValid
FlyingChairsFull = flyingchairs.FlyingChairsFull

## FlyingThings3D_subset
FlyingThings3dFinalTrain = flyingThings3D.FlyingThings3dFinalTrain
FlyingThings3dFinalTest = flyingThings3D.FlyingThings3dFinalTest
FlyingThings3dCleanTrain = flyingThings3D.FlyingThings3dCleanTrain
FlyingThings3dCleanTest = flyingThings3D.FlyingThings3dCleanTest
FlyingThings3dCleanValid = flyingThings3D.FlyingThings3dCleanValid
FlyingThings3dCleanFull = flyingThings3D.FlyingThings3dCleanFull

## Sintel
SintelTestClean = sintel.SintelTestClean
SintelTestFinal = sintel.SintelTestFinal

SintelTrainingCombFull = sintel.SintelTrainingCombFull
SintelTrainingCombTrain = sintel.SintelTrainingCombTrain
SintelTrainingCombValid = sintel.SintelTrainingCombValid

SintelTrainingCleanFull = sintel.SintelTrainingCleanFull
SintelTrainingCleanTrain = sintel.SintelTrainingCleanTrain
SintelTrainingCleanValid = sintel.SintelTrainingCleanValid

SintelTrainingFinalFull = sintel.SintelTrainingFinalFull
SintelTrainingFinalTrain = sintel.SintelTrainingFinalTrain
SintelTrainingFinalValid = sintel.SintelTrainingFinalValid

## Monkaa
MonkaaFinalTrain = monkaa.MonkaaFinalTrain
MonkaaFinalValid = monkaa.MonkaaFinalValid

## KITTI Optical Flow 2012 + 2015
KittiCombTrain = kitti_combined.KittiCombTrain
KittiCombVal = kitti_combined.KittiCombVal
KittiCombFull = kitti_combined.KittiCombFull

KittiComb2012Train = kitti_combined.KittiComb2012Train
KittiComb2012Val = kitti_combined.KittiComb2012Val
KittiComb2012Full = kitti_combined.KittiComb2012Full
KittiComb2012Test = kitti_combined.KittiComb2012Test

KittiComb2015Train = kitti_combined.KittiComb2015Train
KittiComb2015Val = kitti_combined.KittiComb2015Val
KittiComb2015Full = kitti_combined.KittiComb2015Full
KittiComb2015Test = kitti_combined.KittiComb2015Test


## PWCInterfaceDataset
PWCInterfaceDatasetTrain = pwcInterfaceDataset.PWCInterfaceDatasetTrain
PWCInterfaceDatasetValid = pwcInterfaceDataset.PWCInterfaceDatasetValid

## SubsampledDataset
SubsampledDataset = subsampledDataset.SubsampledDataset

## Combined Dataset
CTSKTrain = combined_dataset.CTSKTrain
CTSKValid = combined_dataset.CTSKValid

CTSTrain = combined_dataset.CTSTrain
CTSValid = combined_dataset.CTSValid
