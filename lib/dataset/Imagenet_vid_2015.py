import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from .voc_eval import voc_eval
from .voc_eval import voc_intermittent_eval
from lib.utils.data_augment_RNN import augment_data
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

## UV dataset classes
# CUSTOM_CLASSES = ( '__background__', # always index 0
#     'person', 'car', 'bike', 'obstacle')
# Imagenet vid 2015 classes
CUSTOM_CLASSES = ('__background__',  # always index 0
        'n02691156', 'n02419796', 'n02131653', 'n02834778',
        'n01503061', 'n02924116', 'n02958343', 'n02402425',
        'n02084071', 'n02121808', 'n02503517', 'n02118333',
        'n02510455', 'n02342885', 'n02374451', 'n02129165',
        'n01674464', 'n02484322', 'n03790512', 'n02324045',
        'n02509815', 'n02411705', 'n01726692', 'n02355227',
        'n02129604', 'n04468005', 'n01662784', 'n04530566',
        'n02062744', 'n02391049')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class VOCSegmentation(data.Dataset):

    """VOC Segmentation Dataset Object
    input and target are both images

    NOTE: need to address https://github.com/pytorch/vision/issues/9

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg: 'train', 'val', 'test').
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target image
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_set, transform=None, target_transform=None,
                 dataset_name='VOC2007'):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        self._annopath = os.path.join(
            self.root, dataset_name, 'SegmentationClass', '%s.png')
        self._imgpath = os.path.join(
            self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(
            self.root, dataset_name, 'ImageSets', 'Segmentation', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = Image.open(self._annopath % img_id).convert('RGB')
        img = Image.open(self._imgpath % img_id).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(CUSTOM_CLASSES, range(len(CUSTOM_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0,5)) 
        for obj in target.iter('object'):
            try:
                difficult = int(obj.find('difficult').text) == 1
                if not self.keep_difficult and difficult:
                    continue
            except:
                pass
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class CUSTOMRNNDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets, preproc=None, target_transform=AnnotationTransform(), use_LSTM=False, 
                 batch_size=32, dataset_name='custom', augment_RNN=False):
        self.root = root
        self.image_set = image_sets
        self.preproc = preproc
        self.data_augment_RNN = None
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.test_video_break = []

        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, 'VOC' + year)
            for idx, line in enumerate(open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt'))):
                if name == 'test':
                    if line.strip().rsplit('-')[1] == '000':
                        self.test_video_break.append(idx) #for storing of where videos start and end in test phase
                self.ids.append((rootpath, line.strip()))
        self.ids.sort(key=takeSecond)
        self.augment_data = None
        if augment_RNN:
            self.sample_img, _ = self.__getitem__(0)
            self.augment_data = augment_data(self.ids, self.sample_img)

        


    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.preproc is not None:
            img, target = self.preproc(img, target)
        
        if self.augment_data is not None:
            return self.augment_data(img, target, index)
        else:
            return img, target


    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()

        if self.target_transform is not None:
            anno = self.target_transform(anno)
        return anno
        

    def pull_img_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno)
        height, width, _ = img.shape
        boxes = gt[:,:-1]
        labels = gt[:,-1]
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        labels = np.expand_dims(labels,1)
        targets = np.hstack((boxes,labels))
        
        return img, targets

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def evaluate_detections(self, all_boxes, output_dir=None, printout=True):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        aps,map = self._do_python_eval(output_dir, printout=printout)
        return aps,map

    def evaluate_intermittent_detections(self, boxes, img_id, output_dir=None):

        APs = list()
        empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))
        rootpath = os.path.join(self.root, 'VOC' + self._year)
        name = self.image_set[0][1]
        use_07_metric = True if int(self._year) < 2010 else False
        annopath = os.path.join(
                                rootpath,
                                'Annotations',
                                '{:s}.xml')
        imagesetfile = os.path.join(
                                rootpath,
                                'ImageSets',
                                'Main',
                                name+'.txt')
        for cls_ind, cls in enumerate(CUSTOM_CLASSES):
            if cls == '__background__':
                continue
            rec, prec, ap = voc_intermittent_eval(boxes[cls_ind], annopath, imagesetfile, cls,
                                                ovthresh=0.5,
                                                use_07_metric=use_07_metric,
                                                img_id=self.ids[img_id][1])

            APs.append(ap)
        return APs,np.mean(APs)

    def _get_voc_results_file_template(self):
        filename = 'comp4_det_test' + '_{:s}.txt'
        filedir = os.path.join(
            self.root, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes, printout = True, intermittent=False, img_id=0):
        for cls_ind, cls in enumerate(CUSTOM_CLASSES):
            cls_ind = cls_ind 
            if cls == '__background__':
                continue
            if printout: print('Writing {} CUSTOM results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                if not intermittent:
                    for im_ind, index in enumerate(self.ids):
                        index = index[1]
                        dets = all_boxes[cls_ind][im_ind]
                        if dets == []:
                            continue
                        for k in range(dets.shape[0]):
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(index, dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))
                else:
                    index = self.ids[img_id][1]
                    dets = all_boxes[cls_ind][0]
                    if len(dets) == 0:
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, 0,
                                0, 0,
                                0, 0))
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                dets[k, 0] + 1, dets[k, 1] + 1,
                                dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output', printout=True):
        rootpath = os.path.join(self.root, 'VOC' + self._year)
        name = self.image_set[0][1]
        annopath = os.path.join(
                                rootpath,
                                'Annotations',
                                '{:s}.xml')
        imagesetfile = os.path.join(
                                rootpath,
                                'ImageSets',
                                'Main',
                                name+'.txt')
        cachedir = os.path.join(self.root, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        if printout: print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(CUSTOM_CLASSES):

            if cls == '__background__':
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                                    filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                                    use_07_metric=use_07_metric)
            aps += [ap]
            if printout: print('AP for {} = {:.4f}'.format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        if printout: 
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
            print('-- Thanks, The Management')
            print('--------------------------------------------------------------')
        return aps,np.mean(aps)

    def show(self, index):
        img, target = self.__getitem__(index)
        for obj in target:
            obj = obj.astype(np.int)
            cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (255,0,0), 3)
        cv2.imwrite('./image.jpg', img)


def takeSecond(elem): # for use in sorting
    return elem[1]

class customSampler(data.Sampler):
    def __init__(self, data_source, batch_size = 4, num_frames = 100):
        self.data_source = data_source
        self.imgs = [i for i in range(len(self.data_source))]
        self.vids = []
        self.num_frame = num_frames
        self.batch_size = batch_size
        frame_container = []
        for idx, data in enumerate(self.imgs):
            frame_container.append(data)
            if len(frame_container) == self.num_frame * self.batch_size:
                #each frame_container is a container of batch_size number of videos
                vid = []
                for img in iter(frame_container):
                    #here we take the videos out and put them into self.vids
                    vid.append(img)
                    if len(vid) == self.num_frame:
                        self.vids.append(vid)
                        vid = []
                frame_container = []
        
        #now frame_container is left with the remaining images that don't have enough to make up a batch_size number of videos
        #these data will be ignored in this sampler
        self.shufflevideo()

    def __iter__(self):
        return iter(self.imgs)

    def __len__(self):
        return len(self.data_source)

    def shufflevideo(self):
        # #first we randomly augment the videos, such that for each video, all frames go through the same augment
        # print(torch.tensor(self.vids).size())
        # self.vids = [augment_vid(i) for i in self.vids]
        # here we shuffle the list of videos and convert it back to the img list
        rand_idx = torch.randperm(len(self.vids))
        self.vids = torch.tensor(self.vids)[rand_idx].tolist() #shuffled the video list
        self.imgs = []
        for i in range(int(len(self.vids)/self.batch_size)):
            for frame_idx in range(self.num_frame):
                for ii in range(self.batch_size):
                    self.imgs.append(self.vids[self.batch_size*i+ii][frame_idx])

        #add video augmentation here

class customBatchSampler(data.Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batchsize = batch_size
        self.sampler.data_source.augment_data.resetAugment(self.sampler.vids)
    def __iter__(self):
        batch = []
        self.sampler.shufflevideo()
        self.sampler.data_source.augment_data.resetAugment(self.sampler.vids)
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) ==  self.batchsize:
                yield batch
                batch = []

    def __len__(self):
        return len(self.sampler)//self.batchsize
## test
# if __name__ == '__main__':
#     ds = VOCDetection('../../../../../dataset/VOCdevkit/', [('2012', 'train')],
#             None, AnnotationTransform())
#     print(len(ds))
#     img, target = ds[0]
#     print(target)
#     ds.show(1)