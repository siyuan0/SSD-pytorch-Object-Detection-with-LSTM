from lib.dataset import voc
from lib.dataset import coco
from lib.dataset import custom
from lib.dataset import customRNN

dataset_map = {
                'voc': voc.VOCDetection,
                'coco': coco.COCODetection,
                'custom': custom.CUSTOMDetection,
                'customRNN': customRNN.CUSTOMRNNDetection,
            }

def gen_dataset_fn(name):
    """Returns a dataset func.

    Args:
    name: The name of the dataset.

    Returns:
    func: dataset_fn

    Raises:
    ValueError: If network `name` is not recognized.
    """
    if name not in dataset_map:
        raise ValueError('The dataset unknown %s' % name)
    func = dataset_map[name]
    return func


import torch
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup): #checks for the image which is a tensor
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))): #checks for annotation which is a numpy array
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

from lib.utils.data_augment import preproc
import torch.utils.data as data

def load_data(cfg, phase, cfg_RNN=None):
    if phase == 'train':
        if cfg_RNN is not None:
            # dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TRAIN_SETS,
            #                                  batch_size=cfg.TRAIN_BATCH_SIZE, use_LSTM=True)
            dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TRAIN_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB, useRNN=True),
                                             batch_size=cfg.TRAIN_BATCH_SIZE, use_LSTM=True, augment_RNN=True)
            RNNsampler = customRNN.customSampler(dataset, batch_size=cfg_RNN.BATCH_SIZE, num_frames=cfg_RNN.FRAMES_IN_VIDEO)
            RNNbatchsampler = customRNN.customBatchSampler(sampler=RNNsampler, batch_size=cfg_RNN.BATCH_SIZE)
            data_loader = data.DataLoader(dataset, num_workers=cfg.NUM_WORKERS, batch_sampler=RNNbatchsampler, 
                                            collate_fn=detection_collate, pin_memory=True)
        else:
            dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TRAIN_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, cfg.PROB))
            data_loader = data.DataLoader(dataset, cfg.TRAIN_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=cfg.SHUFFLE, collate_fn=detection_collate, pin_memory=True, drop_last=True)
    if phase == 'eval':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'test':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, -2))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'visualize':
        dataset = dataset_map[cfg.DATASET](cfg.DATASET_DIR, cfg.TEST_SETS, preproc(cfg.IMAGE_SIZE, cfg.PIXEL_MEANS, 1))
        data_loader = data.DataLoader(dataset, cfg.TEST_BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    return data_loader
