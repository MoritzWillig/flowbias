# Analysing and overcoming the dataset bias for optical flow backbone networks

This repository contains the source code for training and evaluating models of the
master's thesis [Analysing and overcoming the dataset bias for optical flow backbone networks](https://moritz-willig.de/projects/flowBias.html).

## Getting started
This code has been developed under Anaconda(Python 3.6), Pytorch 1.1 and CUDA > 9.0 on Ubuntu 18.04.

1. pytorch and tqdm (`conda install -c conda-forge tqdm`)

2. Install the correlation package:
   - Depending on your system, configure `-gencode`, `-ccbin`, `cuda-path` in `models/correlation_package/setup.py` accordingly
   - Then, install the correlation package: `./install.sh`

3. The datasets used for this projects are followings:
    - [FlyingChairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
    - [FlyingThings3D subset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
    - [MPI Sintel Dataset](http://sintel.is.tue.mpg.de/downloads)
    - [KITTI Optical Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

4. Configure paths
    - Copy `config_sample.py` to `config.py`
      - (`config.py` is ignored by git)
    - Adjust the paths for datasets and temp folders to match your setup.
    - Adjust the output paths for the evaluation scripts (`evaluations` folder). 

## Training

The `scripts` folder contains training scripts of experiments demonstrated in the thesis.  
To train the model, you can simply run the script file, e.g., `./pwcnet_no_experts.sh`.  
In script files, please configure your own experiment directory (EXPERIMENTS_HOME). Please also configure the dataset directory (e.g. FLYINGCHAIRS_HOME). Valid values are paths in the local file system or a dataset specified in `dataset_locations` in `config.py`.

## Adding a new model for analysis
The basic error metrics for every trained model are stored separately. For further analysis all evaluation results are collected into a single file `eval_summary.csv`. This step also computes missing values and derived metrics.

To add a new model for analysis, please follow the steps below: 

* Add new line to `evaluations/eval_models.sh`, pointing to the new model(s)
* run `eval_models.sh` (~30min for evaluating a PWC Model on all four datasets)
* open `model_meta.py`
  * add a new line to `model_meta`
  * set the key name to the name chosen in `eval_models.sh`
  * fill in the model parameters
    * the fields correspond to the entries of the `model_meta_fields` array.
  * add the newly inserted key to the model_meta_ordering
    * this is used for ordering models when creating `eval_summary.csv`.
  * run `evaluations/collect_model_results.py` to update the `eval_summary.csv` file.

**Note**: By default `evaluate_for_all_datasets.py` will not reevaluate a dataset split, if the model meta file already contains a result for it. This prevents computing the same results over and over again. If you have changed your evaluation method, use the `reevaluate` variable to force a reevaluation of the affected dataset split(s).


## Info
Author: Moritz Willig (https://moritz-willig.de)  
Base repository: [https://github.com/MoritzWillig/flowbias](https://github.com/MoritzWillig/flowbias)

## Acknowledgement
The repository is based on [Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation](https://github.com/visinf/irr) by Junhwa Hur.

Portions of the source code (e.g., training pipeline, runtime, argument parser, and logger) are from [Jochen Gast](https://www.visinf.tu-darmstadt.de/team_members/jgast/jgast.en.jsp)

