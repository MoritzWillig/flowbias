from flowbias.config import Config
from flowbias.datasets import KittiComb2015Train, KittiComb2015Val, PWCInterfaceDatasetTrain, FlyingThings3dCleanTrain, FlyingThings3dCleanValid

datasets = {
    "kitti2015Train": KittiComb2015Train({}, Config.dataset_locations["kitti"]),
    "kitti2015Valid": KittiComb2015Val({}, Config.dataset_locations["kitti"]),
    "flyingThingsCleanTrain": FlyingThings3dCleanTrain({}, Config.dataset_locations["flyingThings"]),
    "flyingThingsCleanValid": FlyingThings3dCleanValid({}, Config.dataset_locations["flyingThings"])
    #"pwcinterface": PWCInterfaceDatasetTrain({},"/data/dataA/model_interfaces/A_things", "/data/dataA/model_interfaces/A_things")
}


for name, dataset in datasets.items():
    #dataset = KittiComb2015Train({}, "/data/dataB/datasets/KITTI_data_scene_flow/")
    print(name, ":", len(dataset))
