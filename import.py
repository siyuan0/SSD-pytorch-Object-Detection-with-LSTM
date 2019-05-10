import sys
import os
from shutil import copy
import cv2
import  numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
    
# change this settings as required
blank_XML_path = 'blank.xml' # path to a blank xml file for filling up videos to a set number of frames
blank_img_path = 'blank.jpg' # path to a blank image file for filling up videos to a set number of frames
imagenet_path = 'data/ILSVRC2015/' # path to the Imagenet data
output_path = 'data/custom_vid/VOC2007' # path to the folder to store the new processesd data
num_frames = 50 # number of frames in a video. Videos will be filled up/cut to this amount

# This code is meant for converting the Imagenet Vid dataset into one suitabe for training in the SSD net
# This code does the following:
#   1. Transform all frames to standard 300x300 resolution
#   2. Transform the annotations in accordance to how the frames were changed
#   3. Rearrange the video frames in a manner to be read by Imagenet_vid_2015.py when training/testing
# Run this code before running training/testing

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

def copy_transform(orig_img_path, new_img_path, orig_annot_path, new_annot_path, new_dim=(300,300)):
	img = cv2.imread(orig_img_path,cv2.IMREAD_COLOR)
	orig_h = img.shape[0]
	orig_w = img.shape[1]
	img = cv2.resize(img, (new_dim[0], new_dim[1]),interpolation=cv2.INTER_LINEAR)
	

	annot_XML = ET.parse(orig_annot_path)
	annot_XML_tree = annot_XML.getroot()
	for obj in annot_XML_tree.iter('object'):
		bndbox = obj.find('bndbox')
		bndbox.find('xmin').text = str(round(int(bndbox.find('xmin').text) * (new_dim[0]/orig_w)))
		bndbox.find('ymin').text = str(round(int(bndbox.find('ymin').text) * (new_dim[1]/orig_h)))
		bndbox.find('xmax').text = str(round(int(bndbox.find('xmax').text) * (new_dim[0]/orig_w)))
		bndbox.find('ymax').text = str(round(int(bndbox.find('ymax').text) * (new_dim[1]/orig_h)))
		# img = add_bbox(img, int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), 
		# 				int(bndbox.find('xmax').text)-int(bndbox.find('xmin').text), 
		# 				int(bndbox.find('ymax').text)-int(bndbox.find('ymin').text),
		# 				[100,100,100],'blah')

	annot_XML.write(new_annot_path)
	cv2.imwrite(new_img_path, img)



vid_idx = 0
for phase in ('val','train'):
	if phase == 'train':
		imagenet_path_imgfolder = os.path.join(imagenet_path, 'Data', 'VID', phase, 'ILSVRC2015_VID_train_0000')
		imagenet_path_annotfolder = os.path.join(imagenet_path,'Annotations', 'VID', phase, 'ILSVRC2015_VID_train_0000')
	else:
		imagenet_path_imgfolder = os.path.join(imagenet_path, 'Data', 'VID', phase)
		imagenet_path_annotfolder = os.path.join(imagenet_path,'Annotations', 'VID', phase)
	output_path_img = os.path.join(output_path, 'JPEGImages','%s.jpg')
	output_path_annot = os.path.join(output_path, 'Annotations', '%s.xml')

	
	with open(os.path.join(output_path, 'ImageSets', 'Main', phase + '.txt'),'wt') as txt:
		num_files = len(os.listdir(imagenet_path_imgfolder))
		file_counter = 0
		for vid_folder in sorted(os.listdir(imagenet_path_imgfolder)):
			frames = iter(sorted(os.listdir(os.path.join(imagenet_path_imgfolder,vid_folder))))
			for frame_counter in range(num_frames):
				try:
					frame_id = next(frames).rsplit('.')[0]
					frame = os.path.join(imagenet_path_imgfolder, vid_folder,frame_id + '.JPEG')
					annot = os.path.join(imagenet_path_annotfolder, vid_folder, frame_id + '.xml')
				except:
					if phase == 'train':
						frame = blank_img_path
						annot = blank_XML_path
					else:
						break
				new_frame_id = '%s-%s' % (str(vid_idx).zfill(4), str(frame_counter).zfill(3))

				copy_transform(frame, output_path_img % new_frame_id, annot, 
								output_path_annot % new_frame_id, (300,300))
				
				# copy(frame, output_path_img % new_frame_id)
				# copy(annot, output_path_annot % new_frame_id)


				txt.write(new_frame_id +'\n')
			vid_idx += 1
			file_counter += 1
			sys.stdout.write('\rimporting {phase}: {progress}  videos processed:{num}'.format(
							phase=phase,
							progress='#'*int(10*(file_counter/num_files))+'-'*int(10*(1-file_counter/num_files)),
							num=file_counter))
			sys.stdout.flush()
		print('')
