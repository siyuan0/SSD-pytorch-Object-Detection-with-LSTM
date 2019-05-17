from __future__ import print_function
from __future__ import division
import numpy as np
import os
import sys
import cv2
import random
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.nn.init as init

import argparse

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from lib.utils.config_parse import cfg_from_file

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
def add_bbox(image, x1, y1, box_w,box_h, box_color, label):
    #font = ImageFont.truetype("arial.ttf", 18)
    font = ImageFont.load_default()
    #adds the bounding box with label to image
    img_PIL = Image.fromarray(image, mode='RGB')
    draw = ImageDraw.Draw(img_PIL)
    box_color = tuple(int(255*x) for x in box_color)
    draw.line([(x1,y1),(x1+box_w,y1),(x1+box_w,y1+box_h),(x1,y1+box_h),(x1,y1)],
                fill=box_color[0:3], width=5)
    text_size = draw.textsize(str(label),font)
    draw.rectangle((x1,y1,x1+text_size[0],y1+text_size[1]),fill=box_color[0:3])
    draw.text([x1,y1],str(label),fill=(255,255,255),font=font)
    nparr = np.asarray(img_PIL)
    return nparr
def get_data():
	data_loader = load_data(cfg.DATASET, 'train')#, cfg.MODEL.RNN)
	batch_iterator = iter(data_loader)
	initialized_video = False
	video_list = []
	print('collecting images')
	for iteration in range(len(data_loader)):
		try:
			images, targets = next(batch_iterator)
			if images.size()[0] != 32:
				print(iteration)
				print(images.size())
				quit()
		except StopIteration:
			break
		if not initialized_video:
			for i in range(images.size()[0]):
				video_list.append([])
			initialized_video = True
		for i in range(images.size()[0]):
			img = images[i].numpy()
			target = targets[i].numpy()
			
			video_list[i].append((img, target))
	print('printing images to /home/chensy/pythonML/LSTM-SSD-pytorch/trainimages/')
	# print(images.size())
	try:
		os.mkdir('/home/chensy/pythonML/LSTM-SSD-pytorch/trainimages')
	except:
		pass
	for i in range(images.size()[0]):
		# video_writer = cv2.VideoWriter('output_video{d}.avi'.format(d=i)
		# 										,cv2.VideoWriter_fourcc(*'XVID'),12, 
		# 										(video_list[0][0].shape[0],video_list[0][0].shape[1]))
		for ii in range(len(video_list[i])):
			image_out = video_list[i][ii][0].transpose((1,2,0))
			
			# print(image_out.astype(int))
			# quit()
			#add bbox
			target_out = video_list[i][ii][1]

			for bbox in target_out:
				image_out = add_bbox(image_out.astype('uint8'), 300*bbox[0], 300*bbox[1], 300*(bbox[2]-bbox[0]), 300*(bbox[3]-bbox[1]),(0.1,0.5,0.6), bbox[4])
			cv2.imwrite(os.path.join('/home/chensy/pythonML/LSTM-SSD-pytorch/trainimages','{i}-{ii}.jpg'.format(i=i,ii=ii)), image_out)
		

def main():
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    get_data()

if __name__ == '__main__':
    main()