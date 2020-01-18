from argparse import Namespace

from flowbias.datasets.flyingchairs import FlyingChairsTrain, FlyingChairsValid
from flowbias.datasets.flyingThings3D import FlyingThings3dCleanTrain, FlyingThings3dCleanValid
from flowbias.datasets.sintel import SintelTrainingCleanTrain, SintelTrainingCleanValid
from flowbias.datasets.kitti_combined import KittiComb2015Train, KittiComb2015Val
from flowbias.config import Config

class CombinedDataset:

    def __init__(self, args, datasets):
        self._datasets = list(datasets)
        self._size = sum([len(dataset) for dataset in self._datasets])
        self._args = args

    def __getitem__(self, index):
        """
        :param index: (dataset_id, index)
        :return:
        """

        sample_dict = self._datasets[index[0]][index[1]]
        sample_dict["dataset"] = index[0]

        return sample_dict

    def __len__(self):
        return self._size


class CTSKTrain(CombinedDataset):

    def __init__(self, args, photometric_augmentations=True):
        datasets = [
            FlyingChairsTrain(args, Config.dataset_locations["flyingChairs"], photometric_augmentations=photometric_augmentations),
            FlyingThings3dCleanTrain(args, Config.dataset_locations["flyingThings"], photometric_augmentations=photometric_augmentations),
            SintelTrainingCleanTrain(args, Config.dataset_locations["sintel"], photometric_augmentations=photometric_augmentations),
            KittiComb2015Train(args, Config.dataset_locations["kitti"], photometric_augmentations=photometric_augmentations)
        ]
        super().__init__(args, datasets)


class CTSKValid(CombinedDataset):

    def __init__(self, args, photometric_augmentations=True):
        args_kitti = Namespace(**vars(args))
        args_kitti._batch_size = 1
        datasets = [
            FlyingChairsValid(args, Config.dataset_locations["flyingChairs"], photometric_augmentations=photometric_augmentations),
            FlyingThings3dCleanValid(args, Config.dataset_locations["flyingThings"], photometric_augmentations=photometric_augmentations),
            SintelTrainingCleanValid(args, Config.dataset_locations["sintel"], photometric_augmentations=photometric_augmentations),
            KittiComb2015Val(args_kitti, Config.dataset_locations["kitti"], photometric_augmentations=photometric_augmentations)
        ]
        super().__init__(args, datasets)
