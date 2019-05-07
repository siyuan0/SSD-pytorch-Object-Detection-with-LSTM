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
1. install [pytorch](http://pytorch.org/)
2. install requirements by `pip install -r ./requirements.txt` (if this doesn't work, just install the packages in the text file one by one)

## Data
---to continue work here---

## Effect of LSTM
So far it looks like the LSTM layers causes the mAP to drop, while increasing the fps. I haven't really ran this out on the proper imagenet vid dataset, so proper evidence for this is lacking. Just something observed while working on this.