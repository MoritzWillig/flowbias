from flowbias.datasets import KittiComb2015Train, KittiComb2015Val



dataset = KittiComb2015Val({}, "/data/vimb01/KITTI_scene_flow/")
print(">>", len(dataset))
