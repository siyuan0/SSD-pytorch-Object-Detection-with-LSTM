# LSTM-SSD pytorch implementation  

## A note to he who comes after me  
This repo is built on the work of [ssds-pytorch](https://github.com/ShuangXieIrene/ssds.pytorch). What I've added are various pieces of code to test the effect of LSTM layers on the mAP and fps of the model. The LSTM design is based on the [paper](https://arxiv.org/abs/1711.06368) by Mason Liu and Menglong Zhu. To that end, the following files were written:  
1. lib/dataset/customRNN.py  
2. lib/layers/modules/LSTM.py  
3. lib/modeling/ssds/ssd_lite_RNN.py  
4. lib/utils/data_augment_RNN.py  

In addition, various core files were modified to accomodate the inclusion of the LSTM layers, but they should still be able to work for the original models without LSTM.   

## Installation
1. install [pytorch](http://pytorch.org/)
2. install requirements by `pip install -r ./requirements.txt` (if this doesn't work, just install the packages in the text file one by one)

