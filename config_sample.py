
class Config:

    dataset_locations = {
        "flyingChairs": "/data/dataB/datasets/FlyingChairs_release/data/",
        "flyingThings": "/data/dataB/datasets/FlyingThings3D_subset/",
        "sintel": "/data/dataB/datasets/MPI-Sintel-complete/",
        "kitti": "/data/dataB/datasets/KITTI_data_scene_flow/",

        "flyingChairsSample": "/data/dataB/datasets/FlyingChairs_sample402/data/",
        "flyingThingsSample": "/data/dataB/datasets/FlyingThings3D_sample401_subset/",
        "sintelSubset": "/data/dataB/datasets/MPI-Sintel_subset400/",
    }

    temp_directory = "/data/dataB/temp/"
    eval_summary_path = temp_directory + "eval_summary.csv"

    model_directory = "/data/dataB/models/"
