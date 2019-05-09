# LSTM-SSD pytorch implementation  

## A note to he who comes after me  
This repo is built on the work of [ssds-pytorch](https://github.com/ShuangXieIrene/ssds.pytorch). What I've added are various pieces of code to test the effect of LSTM layers on the mAP and fps of the model. The LSTM design is based on the [paper](https://arxiv.org/abs/1711.06368) by Mason Liu and Menglong Zhu. To that end, the following files were written:  
1. lib/dataset/customRNN.py  
2. lib/layers/modules/LSTM.py  
3. lib/modeling/ssds/ssd_lite_RNN.py  
4. lib/utils/data_augment_RNN.py  

In addition, various core files were modified to accomodate the inclusion of the LSTM layers. The modifications were done in such a way that you should still be able to use this repo the same way you would used that from [ssds-pytorch](https://github.com/ShuangXieIrene/ssds.pytorch).  
  
*This repo works best on a linux environment. Windows users may encounter various bugs.  

## Installation
1. Install [pytorch](http://pytorch.org/)
2. Install requirements by `pip install -r ./requirements.txt` (if this doesn't work, just install the packages in the text file one by one)

## Train/Test
To test, run the command `python train.py --cfg=[PATH OF CONFIG FILE]`. For example, testing the baseline SSD Mobilenetv2 is done via `python train.py --cfg=experiments/cfgs/tests/ssd_lite_mobilenetv2_train_baseline.yml`. To train the model instead, change the setting `PHASE: ['test']` to `PHASE: ['train']`.

## Data
The models were trained on a custom dataset. The standard [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset can be used as well. However, if you wish to try out the LSTM layers, use [Imagenet_Vid_2015](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php) dataset instead. I have yet to write a dataloader script to process the Imagenet_Vid_2015 data, so the current stop gap measure is to rearrange that data into VOC format.

## Reults
The mAP of different models are recorded in results_tracking.xlsx. It is observed that adding/removing layers do not significantly affect the mAP, but there is a significant drop in fps of the model when sufficient layers of the base mobilenetv2 are removed. This indicates that majority of the runtime is invested in processing the output of the model's forward  propagation, rather than the forward propagation itself.

## Effect of LSTM
So far it looks like the LSTM layers causes the mAP to drop, while increasing the fps. I haven't really ran this out on the proper imagenet vid dataset, so proper evidence for this is lacking. Just something observed while working on this.