from flowbias.datasets import KittiComb2015Train, KittiComb2015Val, PWCInterfaceDatasetTrain


dataset = PWCInterfaceDatasetTrain({}, "/data/dataA/model_interfaces/A_things", "/data/dataA/model_interfaces/A_things")
#dataset = KittiComb2015Train({}, "/data/dataB/datasets/KITTI_data_scene_flow/")
print(">>", len(dataset))
