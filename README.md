# LSTM-SSD pytorch implementation  

## A note to he who comes after me  
This repo is built on the work of [ssds-pytorch](https://github.com/ShuangXieIrene/ssds.pytorch). What I've added are various pieces of code to test the effect of LSTM layers on the mAP and fps of the model. The LSTM design is based on the [paper](https://arxiv.org/abs/1711.06368) by Mason Liu and Menglong Zhu. To that end, the following files were written:  
1. lib/dataset/customRNN.py  
2. lib/layers/modules/LSTM.py  
3. lib/modeling/ssds/ssd_lite_RNN.py  
4. lib/utils/data_augment_RNN.py  

In addition, various core files were modified to accomodate the inclusion of the LSTM layers. The modifications were done in such a way that you should still be able to use this repo the same way you would used that from [ssds-pytorch](https://github.com/ShuangXieIrene/ssds.pytorch).  
  
*This repo works best on a linux environment. Windows users may encounter various bugs.  

## Docker
Run the docker image by `sh nsh run $(id -u)`. This will activate the docker in the current working directory, under your current user account. In this mode, you will not be able to install new packages, due to the permissions restriction of your current user account. If you wish to install new packages, exit this mode using `exit` and then run the docker image in root account using `sh nsh run root`. In root account, you can install new packages, however, any files you modify would be write-protected under root account. Hence, you may not be able to modify them outside of the docker root account. Once finished installing packages, exit the root acount and run `sh nsh commit` to save the changes you made to the docker environment.

## Installation
1. Install [pytorch](http://pytorch.org/)
2. Install requirements by `pip install -r ./requirements.txt` (if this doesn't work, just install the packages in the text file one by one)

## Train/Test
To test, run the command `python3 train.py --cfg=[PATH OF CONFIG FILE]`. For example, testing the baseline SSD Mobilenetv2 is done via `python3 train.py --cfg=experiments/cfgs/tests/ssd_lite_mobilenetv2_train_baseline.yml`. To train the model instead, change the setting `PHASE: ['test']` to `PHASE: ['train']`.

## Data
The models were trained on a custom dataset. The standard [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset can be used as well. To view the annotated results of the model, in your cfg file, under `TEST`, add `PRINT_IMAGES: True`.

## Results
The mAP of different models are recorded in results_tracking.xlsx. It is observed that adding/removing layers do not significantly affect the mAP, but there is a significant drop in fps of the model when sufficient layers of the base mobilenetv2 are removed. This indicates that majority of the runtime is invested in processing the output of the model's forward  propagation, rather than in the forward propagation itself. The results are documented in results_tracking.xlsx, under 'Cutting layers from model' tab.

## Effect of LSTM
So far it looks like the LSTM layers causes the mAP to drop, while increasing the fps. This effect was observed when running it on the IR dataset.

## Pruning
Additional work was done to prune the model, under `prune.py`. The idea is from 'Pruning Filters for Efficient ConvNets' by Hao Li, et al (https://arxiv.org/abs/1608.08710). To use it in your code, add this line `model = prune_model(model, factor_removed=[PROPORTION OF CHANNELS TO REMOVE])`. The model will be pruned based on its current weights, so only do this after completing training. The results are documented under 'pruning conv weights by channel' tab.  


In short, the pruning of up to 50% of the parameters, through pruning by conv2d channels, resulted in negligible decrease in mAP and fps of the model. This can be a good way to simplify models.  

A seperate public repo on pruning is set up at (https://github.com/siyuan0/pytorch_model_prune).

## Tensorboard
To view tensorboard on a remote computer, add this line to your code when running `tensorboard --logdir=[LOG DIRECTORY] & `. One example is `tensorboard --logdir=experiments/models/ssd_mobilenet_v2_custom0 & python3 train.py --cfg=experiments/cfgs/tests/ssd_lite_mobilenetv2_train_baseline.yml`. This will start the tensorboard session alongside your training/testing. Doing this inside docker will set the output to port 5002. On your remote computer, open command prompt and run `ssh chensy@202.115.31.47 -L 5002:127.0.0.1:5002 -p 13022`. Lastly, open your browser and enter the url "http://127.0.0.1:5002/" to view the tensorboard.