from flowbias.config import Config
from flowbias.datasets import KittiComb2015Train, KittiComb2015Val, PWCInterfaceDatasetTrain, FlyingThings3dCleanTrain, \
    FlyingThings3dCleanValid, SintelTrainingCleanTrain, FlyingChairsTrain
from flowbias.utils.meta_infrastructure import get_available_datasets

datasets = get_available_datasets()
"""
datasets = {
    "flyingChairsTrain": FlyingChairsTrain({}, Config.dataset_locations["flyingChairs"]),
    "kitti2015Train": KittiComb2015Train({}, Config.dataset_locations["kitti"]),
    "kitti2015Valid": KittiComb2015Val({}, Config.dataset_locations["kitti"]),
    "flyingThingsCleanTrain": FlyingThings3dCleanTrain({}, Config.dataset_locations["flyingThings"]),
    "flyingThingsCleanValid": FlyingThings3dCleanValid({}, Config.dataset_locations["flyingThings"]),
    "sintelTrainingCleanTrain": SintelTrainingCleanTrain({}, Config.dataset_locations["sintel"])
    #"pwcinterface": PWCInterfaceDatasetTrain({},"/data/dataA/model_interfaces/A_things", "/data/dataA/model_interfaces/A_things")
}"""


for name, dataset in datasets.items():
    print(name, ":", len(dataset))
