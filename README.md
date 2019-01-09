<p align="center"><img width="50%" src="logo/Pytorch_logo.png" /></p>

# U-Net
Pytorch implementation of U-Net

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Build Status](https://travis-ci.com/mukeshmithrakumar/U-Net.svg?branch=master)](https://travis-ci.com/mukeshmithrakumar/U-Net)
[![Coverage Status](https://coveralls.io/repos/github/mukeshmithrakumar/UNet/badge.svg?branch=master)](https://coveralls.io/github/mukeshmithrakumar/UNet?branch=master)

### If this repository helps you in anyway, show your love :heart: by putting a :star: on this project :v: 
[![](https://img.shields.io/github/stars/mukeshmithrakumar/U-Net.svg?label=Stars&style=social)](https://github.com/mukeshmithrakumar/U-Net/stargazers)
[![HitCount](http://hits.dwyl.io/mukeshmithrakumar/U-Net.svg)](http://hits.dwyl.io/mukeshmithrakumar/U-Net)

### For any questions and collaborations you can reach me via: [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue.svg?logo=#0077B5)](https://www.linkedin.com/in/mukesh-mithrakumar/)

## (Work in Progress)
The __Goal__ of the project is to develop an open source software package to assist radiologists in the evaluation of 
lesions in CT and MRI scans for multiple organs.

Currently the code works for the ISBI Neuronal Stack Segmentation dataset. 
See upcoming updates for more info

### Folder Structure

```
main_dir
    - data (The folder containing data files for training and testing)
    - pytorch_unet (Package directory)
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
            - visualize (will be created)
            - weights (will be created)
            - evaluate.py
            - interpret.py
            - train.py
        - utils
            - helpers.py
            - metrics.py
            - proto.py
            - unit_test.py
        - visualize
            - logger.py
            - plot.py
```

## Usage

### 1. Train Mode
To train the model run:
```
task.py root_dir(path/to/root directory)
```

Arguments that can be specified in the training mode:
```
usage: train.py [-h] [--main_dir MAIN_DIR] [--resume]
                [--weights_dir WEIGHTS_DIR] [--log_dir LOG_DIR]
                [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE]
                [--epochs EPOCHS] [--depth DEPTH] [--n_classes N_CLASSES]
                [--up_mode {upconv, upsample}] [--augment]
                [--augment_type {geometric, image, both}]
                [--transform_prob TRANSFORM_PROB] [--test_size TEST_SIZE]
                [--log] [--build_graph]

Script for training the model

optional arguments:
  -h, --help            show this help message and exit
  --main_dir MAIN_DIR   main directory
  --resume              Choose to start training from checkpoint
  --weights_dir WEIGHTS_DIR
                        Choose directory to save weights model
  --log_dir LOG_DIR     Choose directory to save the logs
  --image_size IMAGE_SIZE
                        resize image size
  --batch_size BATCH_SIZE
                        batch size
  --epochs EPOCHS
  --depth DEPTH         Number of downsampling/upsampling blocks
  --n_classes N_CLASSES
                        Number of classes in the dataset
  --up_mode {upconv, upsample}
                        Type of upsampling
  --augment             Whether to augment the train images or not
  --augment_type {geometric, image, both}
                        Which type of augmentation to choose from: geometric,
                        brightness or both
  --transform_prob TRANSFORM_PROB
                        Probability of images to augment when calling
                        augmentations
  --test_size TEST_SIZE
                        Validation size to split the data, should be in
                        between 0.0 to 1.0
  --log                 Log the Values
  --build_graph         Build the model graph
```


#### Logging 
To activate logging of the errors (:default is set as no)
```
task.py root_dir(path/to/root directory) --log yes
```

To see the log in tensorboard follow the log statement after training:

<img src="logo/histogram_logs2.gif">

#### Network Graph
Since Pytorch graphs are dynamic I couldn't yet integrate it with thensorflow but as a quick hack run the following
to build a png version of the model architecture (:default is set as no)
```
task.py root_dir(path/to/root directory) --build_graph yes
```

<img src="logo/u_net_model.png">

### 2. Test Mode
To evaluate the model on the test data run:
```
task.py root_dir(path/to/root directory) --mode evaluate
```

Arguments that can be specified in the eval mode:
```
usage: evaluate.py [-h] [--main_dir MAIN_DIR] [--image_size IMAGE_SIZE]
                   [--weights_dir WEIGHTS_DIR]

Script for evaluating the trained model

optional arguments:
  -h, --help            show this help message and exit
  --main_dir MAIN_DIR   main directory
  --image_size IMAGE_SIZE
                        resize image size to match train image size
  --weights_dir WEIGHTS_DIR
                        Choose directory to save weights model
```


### 3. Interpret Mode
To visualize the intermediate layers:
```
task.py root_dir(path/to/root directory) --mode interpret
```

Arguments that can be specified in the interpret mode:
```
usage: interpret.py [-h] [--main_dir MAIN_DIR]
                    [--interpret_path INTERPRET_PATH]
                    [--weights_dir WEIGHTS_DIR] [--image_size IMAGE_SIZE]
                    [--depth DEPTH]
                    [--plot_interpret {sensitivity,block_filters}]
                    [--plot_size PLOT_SIZE]

Script for interpreting the trained model results

optional arguments:
  -h, --help            show this help message and exit
  --main_dir MAIN_DIR   main directory
  --interpret_path INTERPRET_PATH
                        Choose directory to save layer visualizations
  --weights_dir WEIGHTS_DIR
                        Choose directory to load weights from
  --image_size IMAGE_SIZE
                        resize image size
  --depth DEPTH         Number of downsampling/upsampling blocks
  --plot_interpret {sensitivity,block_filters}
                        Type of interpret to plot
  --plot_size PLOT_SIZE
                        Image size of sensitivity analysis
```


#### Sensitivity Analysis
Is the default option when you run interpret mode
<img src="logo/sensitivity.png">

#### Block Analysis
To visualize the weight output of each downsampling block run:
```
task.py root_dir(path/to/root directory) --mode interpret --plot_interpret block_filters
```
<p align="center"><img width="80%" src="logo/filters.gif" /></p>


## Keep an eye out :eyes: for Upcoming Updates [![](https://img.shields.io/github/watchers/mukeshmithrakumar/U-Net.svg?label=Watch&style=social)](https://github.com/mukeshmithrakumar/U-Net/watchers)
- write performance.py to measure code performance and optimize code
- work on the hyperparamters.py to tune hyper parameters
- write config.py
- add multi gpu capabilities
- write unit_test.py for the above
- add code coverage to check tests and iterate
- Deploy pre alpha PyPI package
- work on a biomedical image pre-processing script
- modify the unet to work on MRI data
- test on the CHAOS Segmentation challenge
- modify the unet to work on CT scan
- test on the PAVES Segmentation challenge
- write unit_test.py for the above
- write a neural architecture search script
- Build a classifier to identify between the organs (One U-Net to segment different organs) 
- Build another separate classifier to identify different cells
- Deploy alpha PyPI package
- Build a graphical user interface for radnet
- Build a developer and researcher mode for the GUI
- Abstract away the deep learning stuff so its not python/deep learning friendly but more like doctor friendly
- Build into a software package
- Deploy beta PyPI package
