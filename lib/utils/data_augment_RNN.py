"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import torch
import cv2
import numpy as np
import random
import math
from lib.utils.box_utils import matrix_iou

RGB_FILL = (103.94, 116.78, 123.68)

def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes)== 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3,1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])
            
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t,labels_t


def _distort(image, rand_set):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()


    _convert(image, beta=rand_set[0])

    _convert(image, alpha=rand_set[1])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    tmp = image[:, :, 0].astype(int) + rand_set[2]
    tmp %= 180
    image[:, :, 0] = tmp

    _convert(image[:, :, 1], alpha=rand_set[3])

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, scale_list=1, ratio_rand_list=0, left_scale_list=0, top_scale_list=0):
    fill = RGB_FILL #to fill the places where image is gone after expand
    height, width, depth = image.shape
    for idx in range(50):
        scale = scale_list[idx]
        ratio_rand = ratio_rand_list[idx]
        left_scale = left_scale_list[idx]
        top_scale = top_scale_list[idx]

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(ratio_rand * (max_ratio-min_ratio) + min_ratio)
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = int(left_scale*(w - width))
        top = int(top_scale*(h - height))

        # #test__
        # w = 400
        # h = 400
        # left = 50
        # top = 50
        # #___

        boxes_t = boxes.copy()

        boxes_t[:,0] = boxes_t[:,0] * width + left
        boxes_t[:,1] = boxes_t[:,1] * height + top
        boxes_t[:,2] = boxes_t[:,2] * width + left
        boxes_t[:,3] = boxes_t[:,3] * height + top

        # boxes_t[:, :2] = boxes_t[:, :2] * 300 + (left, top)
        # boxes_t[:, 2:] = boxes_t[:, 2:] * 300 + (left, top)

        expand_image = np.empty(
            (h, w, depth),
            dtype=np.float32)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image

        image = expand_image
        boxes_t[:,0::2] /= w
        boxes_t[:,1::2] /= h
        # cv2.imshow('image',image)
        image = cv2.resize(image, (height, width),interpolation=cv2.INTER_LINEAR)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # quit()
        image -= RGB_FILL
        
    
        return image, boxes_t


def _mirror(image, boxes):
    #something wrong here, give up for now
    _, width, _ = image.shape

    image = image[::-1,:]
    boxes_t = boxes.copy()
    boxes_t[:, 0::2] = 1 - boxes_t[:, 2::-2]
    return image, boxes


def _elastic(image, elastic_blurshape, alpha=None, sigma=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
     From: 
     https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    """
    if alpha == None:
        alpha = image.shape[0] * random.uniform(0.5,2)
    if sigma == None:
        sigma = int(image.shape[0] * random.uniform(0.5,1))

    shape = image.shape[:2]
    
    dx, dy = [cv2.GaussianBlur((elastic_blurshape * 2 - 1) * alpha, (sigma|1, sigma|1), 0) for _ in range(2)]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = np.clip(x+dx, 0, shape[1]-1).astype(np.float32), np.clip(y+dy, 0, shape[0]-1).astype(np.float32)
    return cv2.remap(np.float32(image), x, y, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT)

class augment_data():
    def __init__(self, img_idx_list, sample_img):
        self.augment_states = [] #records a list of augments for each video
        self.img_augment_match = [0 for _ in range(len(img_idx_list))]
        self.img_augment_match_initialized = False
        self.sample_img = sample_img.transpose(0,2).numpy()

    def __call__(self, image, target, img_idx):
        # finds the correct augment for the image
        if self.img_augment_match_initialized:
            return augment_img(image, target, self.augment_states[self.img_augment_match[img_idx]])
        else:
            return image

    def resetAugment(self,vid_idx_list):
        self.augment_states = []
        for idx, vid in enumerate(vid_idx_list):
            random_state = np.random.RandomState(None)
            augment_state = { #creating a specific augment for a specific video
                'elastic_blurshape': random_state.rand(*self.sample_img.shape[:2]),
                'elastic_alpha': self.sample_img.shape[0] * random.uniform(0.5,2),
                'elastic_sigma': int(self.sample_img.shape[0] * random.uniform(0.5,1)),
                'dropout mask': random_state.rand(*self.sample_img.shape) > 0.2,
                'distort randset': [random.uniform(-32, 32),
                                    random.uniform(0.5, 1.5),
                                    random.randint(-18, 18),
                                    random.uniform(0.5, 1.5)],
                'mirror_do': random.randrange(2),
                'expand_do': random.randrange(2),
                'expand_scale': [random.uniform(1,4) for _ in range(50)],
                'expand_ratio_rand': [random.random() for _ in range(50)],
                'expand_left_scale_list': [random.random() for _ in range(50)],
                'expand_top_scale_list': [random.random() for _ in range(50)]

            }
            self.augment_states.append(augment_state)
            for image_idx in vid:
                self.img_augment_match[image_idx] = idx #store the particular augment for each image
        self.img_augment_match_initialized = True

def augment_img(image, target, augment_state):

    image_t = image.transpose(0,2).numpy()
    boxes_t = target[:,:-1]
    labels_t = target[:,-1]
    #elastic settings
    elastic_blurshape = augment_state['elastic_blurshape']
    elastic_alpha = augment_state['elastic_alpha']
    elastic_sigma = augment_state['elastic_sigma']
    #dropout settings
    dropout_mask = augment_state['dropout mask']
    #distort settings
    distort_randset = augment_state['distort randset']
    #mirror settings
    mirror_do = augment_state['mirror_do']
    #expand settings
    expand_do = augment_state['expand_do']
    expand_scale = augment_state['expand_scale']
    expand_ratio_rand = augment_state['expand_ratio_rand']
    expand_left_scale_list = augment_state['expand_left_scale_list']
    expand_top_scale_list = augment_state['expand_top_scale_list']
    #elastic
    image_t = _elastic(image_t, elastic_blurshape, alpha=elastic_alpha, sigma=elastic_sigma)
    
    #distort
    image_t = _distort(image_t, distort_randset)

    #dropout
    image_t = dropout_mask * image_t

    #mirror
    # if mirror_do:
    # image_t, boxes_t = _mirror(image_t, boxes_t)
    #expand
    if expand_do:
        image_t, boxes_t = _expand(image_t, boxes_t, expand_scale, expand_ratio_rand, expand_left_scale_list, expand_top_scale_list)

    

    labels_t = np.expand_dims(labels_t,1)
    target_t = np.hstack((boxes_t,labels_t))

    return torch.from_numpy(image_t.copy()).transpose(0,2), target_t

