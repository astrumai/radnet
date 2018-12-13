<p align="center"><img width="40%" src="logo/Pytorch_logo.png" /></p>

--------------------------------------------------------------------------------
# U-Net
Pytorch implementation of U-Net

[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Build Status](https://travis-ci.com/mukeshmithrakumar/U-Net.svg?branch=master)](https://travis-ci.com/mukeshmithrakumar/U-Net)
[![HitCount](http://hits.dwyl.io/mukeshmithrakumar/U-Net.svg)](http://hits.dwyl.io/mukeshmithrakumar/U-Net)

### If this repository helps you in anyway, show your love :heart: by putting a :star: on this project :v:

## (Work in Progress)

The __Goal__ of the project is to develop a generalizable model to assist in the evaluation of lesions 
(e.g. benign and malignant tumors, multiple sclerosis and cysts) in CT and MRI scans for multiple organs.

Currently the code works for the ISBI Neuronal Stack Segmentation dataset. 
See Future Works for upcoming updates

### Folder Structure

```
root_dir
    - data (The folder containing data files for training and testing)
    - model (PyTorch u-net model)
        - u_net.py
    - optimize
        - config.py
        - hyperparameter.py
    - processing
        - augments.py
        - load.py
    - trainer
        - train_logs (will be created)
        - weights (will be created)
        - evaluate.py
        - task.py
        - train.py
    - utils
        - helpers.py
        - metrics.py
        - proto.py
        - unit_test.py
    - visualize
        - interpret.py
        - logger.py
        - plot.py
```

## Training
Run the ```task.py``` from the trainer folder. 

#### Usage
```
task.py root_dir(path/to/root directory)
```

## Testing
To evaluate the model on test data run ```task.py root_dir(path/to/root directory) --mode evaluate``` for evaluation.

#### Usage
```
convert_model.py main_dir(path/to/main directory) model_in(model name to be used to convert)
evaluate.py main_dir(path/to/main directory) model_in(model name to be used for evaluation)
```

### Future Plan
- write the logger.py to log values and visualize in tensorboard
- write plot.py into train and plot the logged values
- work on augmentations (it is not working yet)
- finish up the evaluation.py to export data
- work on the hyperparamters.py to write a script to tune hyperparameters
- modify the unet to work on the CHAOS Segmentation challenge
- modify the unet to work on the PAVES Segmentation challenge
- modify the unet to work on the Histopathologic cancer classification challenge
- work on the visualize.py to visualize intermediate layers and build interpretability
- explore possibilities of converting the tensorflow capsnet to pytorch capsnet
- run the capsnet on the above challenges
- write a visualization for capsnet
- compare unet and capsnet
- write the paper on visualizing models for biomedical image segmentation

