import cv2
import os
import argparse
import numpy as np
import sys

def img_to_vid(folderpath):
	videomake = None
	total_img = 0
	img_path_list = []
	for filename in os.listdir(folderpath):
		img_path_list.append(os.path.join(folderpath, filename))
		if videomake is None:
			img_path = os.path.join(folderpath, filename)
			img = np.array(cv2.imread(img_path, flags=1))
			sample_shape = img.shape
			videomake = cv2.VideoWriter('output_video2.avi',cv2.VideoWriter_fourcc(*'XVID'),12, (sample_shape[1], sample_shape[0]))
		total_img += 1
	img_path_list.sort()
	counter = 0
	for img_path in iter(img_path_list):
		img = np.array(cv2.imread(img_path, flags=1))
		videomake.write(img)
		counter += 1
		log = '\rWorking on frames {counter:d}/{total_img:d} |{progress}|'.format(
				counter = counter, total_img = total_img, 
				progress = '#'*int(round(10*counter/total_img)) + '-'*(10-int(round(10*counter/total_img))))
		sys.stdout.write(log)
		sys.stdout.flush()
	videomake.release()
	print('')

def vid_to_img(filepath):
	print("hello world")


class vid_frame_loader():
	def __init__(self, video_path):
		self.video_path = video_path


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='convert vid to img and vice versa')
    parser.add_argument('--org', dest='original_type',
            help='original type of file', choices = ['img','vid'], default=None, type=str)
    parser.add_argument('--filepath', dest='filepath',
            help='filepath/folderpath', default=os.getcwd(), type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
	args = parse_args()
	if args.original_type == 'img':
		img_to_vid(args.filepath)
	elif args.original_type == 'vid':
		vid_to_img(args.filepath)

