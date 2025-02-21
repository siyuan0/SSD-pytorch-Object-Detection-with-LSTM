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

from tensorboardX import SummaryWriter

from lib.layers import *
from lib.utils.timer import Timer
from lib.utils.data_augment import preproc
from lib.modeling.model_builder import create_model
from lib.dataset.dataset_factory import load_data
from lib.utils.config_parse import cfg
from lib.utils.eval_utils import *
from lib.utils.visualize_utils import *
from lib.layers.modules.LSTM import reset_model_LSTM, trigger_TBPTT_model_LSTM
import prune
from prune import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

IMAGE_ID = 0

def print_detections(images, detections, folder='/home/chensy/pythonML/LSTM-SSD-pytorch/trainimages'):
    global IMAGE_ID
    def add_bbox(image, x1, y1, x_max,y_max, box_color, label):
        #font = ImageFont.truetype("arial.ttf", 18)
        font = ImageFont.load_default()
        #adds the bounding box with label to image
        img_PIL = Image.fromarray(image, mode='RGB')
        draw = ImageDraw.Draw(img_PIL)
        box_color = tuple(int(255*x) for x in box_color)
        draw.line([(x1,y1),(x_max,y1),(x_max,y_max),(x1,y_max),(x1,y1)],
                    fill=box_color[0:3], width=5)
        text_size = draw.textsize(str(label),font)
        draw.rectangle((x1,y1,x1+text_size[0],y1+text_size[1]),fill=box_color[0:3])
        draw.text([x1,y1],str(label),fill=(255,255,255),font=font)
        nparr = np.asarray(img_PIL)
        return nparr
    try:
        os.mkdir(folder)
    except:
        pass
    for idx in range(images.size()[0]):
        image = images[idx].cpu().numpy().transpose((1,2,0))
        scale = np.array([image.shape[0], image.shape[1],image.shape[0], image.shape[1]])
        detection = detections[idx]
        for cls_idx in range(5):
            for det in detection[cls_idx]:
                if det[0] > 0.6:
                    box = det[1:].cpu().numpy() * scale
                    image = add_bbox(image.astype('uint8'), box[0], box[1], box[2], box[3], (0.1,0.5,0.6), cls_idx)
        cv2.imwrite(os.path.join(folder,'{i}.jpg'.format(i=IMAGE_ID)), image)
        IMAGE_ID += 1
        # quit()

class Solver(object):
    """
    A wrapper class for the training process
    """
    def __init__(self):
        self.cfg = cfg

        # Build model
        print('===> Building model')
        self.model, self.priorbox = create_model(cfg.MODEL)
        print(self.model)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.detector = Detect(cfg.POST_PROCESS, self.priors)

        # Load data
        print('===> Loading data')
        self.train_loader = load_data(cfg.DATASET, 'train', cfg.MODEL.RNN) if 'train' in cfg.PHASE and cfg.MODEL.RNN.IN_USE else None
        self.train_loader_noLSTM = load_data(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
        self.eval_loader = load_data(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        self.test_loader = load_data(cfg.DATASET, 'test') #if 'test' in cfg.PHASE else None
        self.visualize_loader = load_data(cfg.DATASET, 'visualize') if 'visualize' in cfg.PHASE else None

        self.RNN_in_use = cfg.MODEL.RNN.IN_USE
        self.backprop_steps = cfg.MODEL.RNN.BACKPROP_STEPS if self.RNN_in_use else False #for RNN use, to track how far back to backprop
        self.frames_in_video = cfg.MODEL.RNN.FRAMES_IN_VIDEO if self.RNN_in_use else False #for resetting the LSTM
        if cfg.TEST.VIDEO_BREAK == ['from_dataset']:
            self.test_video_break = self.test_loader.dataset.test_video_break
        else:
            self.test_video_break = cfg.TEST.VIDEO_BREAK


        # Utilize GPUs for computation
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print('Utilize GPUs for computation')
            for GPU_ID in range(torch.cuda.device_count()):
                print('Using GPU: %s' %(torch.cuda.get_device_name(GPU_ID)))
            self.model.cuda()
            self.priors.cuda()
            cudnn.benchmark = True
           

        # print trainable scope
        print('Trainable scope: {}'.format(cfg.TRAIN.TRAINABLE_SCOPE))
        trainable_param = self.trainable_param(cfg.TRAIN.TRAINABLE_SCOPE)
        self.optimizer = self.configure_optimizer(trainable_param, cfg.TRAIN.OPTIMIZER)
        self.exp_lr_scheduler = self.configure_lr_scheduler(self.optimizer, cfg.TRAIN.LR_SCHEDULER)
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS

        # metric
        self.criterion = MultiBoxLoss(cfg.MATCHER, self.priors, self.use_gpu)

        # Set the logger
        self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)
        self.output_dir = cfg.EXP_DIR
        self.checkpoint = cfg.RESUME_CHECKPOINT
        self.checkpoint_prefix = cfg.CHECKPOINTS_PREFIX


    def save_checkpoints(self, epochs, iters=None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if iters:
            filename = self.checkpoint_prefix + '_epoch_{:d}_iter_{:d}'.format(epochs, iters) + '.pth'
        else:
            filename = self.checkpoint_prefix + '_epoch_{:d}'.format(epochs) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('epoch {epoch:d}: {filename}\n'.format(epoch=epochs, filename=filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        # TODO: write relative cfg under the same page

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        # print("=> Weigths in the checkpoints:")
        # print([k for k, v in list(checkpoint.items())])

        # remove the module in the parrallel model
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        resume_scope = self.cfg.TRAIN.RESUME_SCOPE
        # extract the weights based on the resume scope
        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in self.model.state_dict()}

        checkpoint = self.model.state_dict()

        unresume_dict = set(checkpoint)-set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        checkpoint.update(pretrained_dict)

        return self.model.load_state_dict(checkpoint)


    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('epoch ') + len('epoch '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def weights_init(self, m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0


    def initialize(self):
        # TODO: ADD INIT ways
        # raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")
        # for module in self.cfg.TRAIN.TRAINABLE_SCOPE.split(','):
        #     if hasattr(self.model, module):
        #         getattr(self.model, module).apply(self.weights_init)
        if self.checkpoint:
            print('Loading initial model weights from {:s}'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)

        start_epoch = 0
        return start_epoch

    def trainable_param(self, trainable_scope):
        for param in self.model.parameters():
            param.requires_grad = False

        trainable_param = []
        for module in trainable_scope.split(','):
            if hasattr(self.model, module):
                # print(getattr(self.model, module))
                for param in getattr(self.model, module).parameters():
                    param.requires_grad = True
                trainable_param.extend(getattr(self.model, module).parameters())

        return trainable_param

    def train_model(self):
        previous = self.find_previous()
        if previous:
            start_epoch = previous[0][-1]
            self.resume_checkpoint(previous[1][-1])
        else:
            start_epoch = self.initialize()

        # export graph for the model, onnx always not works
        # self.export_graph()

        # warm_up epoch
        warm_up = self.cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
        for epoch in iter(range(start_epoch+1, self.max_epochs+1)):
            #learning rate
            sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.max_epochs))
            if epoch > warm_up:
                self.exp_lr_scheduler.step(epoch-warm_up)
            if 'train' in cfg.PHASE:
                self.model = reset_model_LSTM(self.model)

                if epoch < self.cfg.MODEL.RNN.USE_LSTM_AFTER_EPOCH or not self.cfg.MODEL.RNN.IN_USE:
                    with modules.LSTM.no_LSTM(): # train the first few epoch without any use of LSTM
                        self.train_epoch(self.model, self.train_loader_noLSTM, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu)
                        if self.cfg.TRAIN.TRACK_MAP and epoch%self.cfg.TRAIN.TRACK_MAP_EVERY==0 and epoch!=1:
                            self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu, self.writer, epoch, printout=False)
                else:
                    self.train_epoch(self.model, self.train_loader, self.optimizer, self.criterion, self.writer, epoch, self.use_gpu, use_RNN=self.RNN_in_use)
                    if self.cfg.TRAIN.TRACK_MAP and epoch%self.cfg.TRAIN.TRACK_MAP_EVERY==0 and epoch!=1:
                        self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu, self.writer, epoch, printout=False)
            if 'eval' in cfg.PHASE:
                self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
            if 'test' in cfg.PHASE:
                self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir, self.use_gpu, self.writer)
            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)

            if epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
                self.save_checkpoints(epoch)

    def test_model(self):
        previous = self.find_previous()
        if previous:
            for epoch, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= epoch <= self.cfg.TEST.TEST_SCOPE[1]:
                    sys.stdout.write('\rEpoch {epoch:d}/{max_epochs:d}:\n'.format(epoch=epoch, max_epochs=self.cfg.TEST.TEST_SCOPE[1]))
                    self.resume_checkpoint(resume_checkpoint)
                    if 'eval' in cfg.PHASE:
                        self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, epoch, self.use_gpu)
                    if 'test' in cfg.PHASE:
                        self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
                    if 'visualize' in cfg.PHASE:
                        self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, epoch,  self.use_gpu)
        else:
            sys.stdout.write('\rCheckpoint {}:\n'.format(self.checkpoint))
            self.resume_checkpoint(self.checkpoint)
            if 'eval' in cfg.PHASE:
                self.eval_epoch(self.model, self.eval_loader, self.detector, self.criterion, self.writer, 0, self.use_gpu)
            if 'test' in cfg.PHASE:
                self.test_epoch(self.model, self.test_loader, self.detector, self.output_dir , self.use_gpu)
            if 'visualize' in cfg.PHASE:
                self.visualize_epoch(self.model, self.visualize_loader, self.priorbox, self.writer, 0,  self.use_gpu)


    def train_epoch(self, model, data_loader, optimizer, criterion, writer, epoch, use_gpu, use_RNN=False):
        model.train()
        print('dataset size: %d' %len(data_loader.dataset))
        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        out_history = []
        _t = Timer()
        RNN_timesteps = 0

        for iteration in iter(range((epoch_size))):
            
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                break

            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            _t.tic()
            # forward
            
            out = model(images, phase='train')
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c

            if use_RNN: 
                # out_history.append([out, targets])
                RNN_timesteps += 1
                next_video = ((iteration+1)%self.frames_in_video == 0) #check if gonna start next videoset for training
                if RNN_timesteps == self.backprop_steps or iteration == epoch_size or next_video:
                    #only optimize after a number of timesteps
                    optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    trigger_TBPTT_model_LSTM(model, 0) #calls backprop for the past cell states
                    optimizer.step()
                    RNN_timesteps = 0
                    # out_history = []
                # optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                # trigger_TBPTT_model_LSTM(model,-1)
                # optimizer.step()
                if next_video:
                    #resets the model's LSTM so that next batch can be treated as a new set of videos
                    with torch.no_grad():
                        reset_model_LSTM(model)
            else:
                loss_l, loss_c = criterion(out, targets)
                # some bugs in coco train2017. maybe the annonation bug.
                if loss_l.item() == float("Inf"):
                    continue
                # loss = loss_l + loss_c
                optimizer.zero_grad()
                loss = loss_l + loss_c #putting more pressure on the location loss, cos loss_c converges too fast and tend to explode
                loss.backward()
                optimizer.step()
                

            time = _t.toc()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            # log per iter
            log = '\r==>Train: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.item(), cls_loss=loss_c.item())

            sys.stdout.write(log)
            sys.stdout.flush()

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        lr = optimizer.param_groups[0]['lr']
        log = '\r==>Train: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || lr: {lr:.6f}\n'.format(lr=lr,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Train/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Train/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Train/lr', lr, epoch)


    def eval_epoch(self, model, data_loader, detector, criterion, writer, epoch, use_gpu):
        model.eval()

        epoch_size = len(data_loader)
        batch_iterator = iter(data_loader)

        loc_loss = 0
        conf_loss = 0
        _t = Timer()

        label = [list() for _ in range(model.num_classes)]
        gt_label = [list() for _ in range(model.num_classes)]
        score = [list() for _ in range(model.num_classes)]
        size = [list() for _ in range(model.num_classes)]
        npos = [0] * model.num_classes

        for iteration in iter(range((epoch_size))):
        # for iteration in iter(range((10))):
            images, targets = next(batch_iterator)
            if use_gpu:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            _t.tic()
            # forward
            out = model(images, phase='train')

            # loss
            loss_l, loss_c = criterion(out, targets)

            out = (out[0], model.softmax(out[1].view(-1, model.num_classes)))

            # detect
            detections = detector.forward(out)

            time = _t.toc()

            # evals
            label, score, npos, gt_label = cal_tp_fp(detections, targets, label, score, npos, gt_label)
            size = cal_size(detections, targets, size)
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            # log per iter
            log = '\r==>Eval: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] || loc_loss: {loc_loss:.4f} cls_loss: {cls_loss:.4f}\r'.format(
                    prograss='#'*int(round(10*iteration/epoch_size)) + '-'*int(round(10*(1-iteration/epoch_size))), iters=iteration, epoch_size=epoch_size,
                    time=time, loc_loss=loss_l.data[0], cls_loss=loss_c.data[0])

            sys.stdout.write(log)
            sys.stdout.flush()

        # eval mAP
        prec, rec, ap = cal_pr(label, score, npos)

        # log per epoch
        sys.stdout.write('\r')
        sys.stdout.flush()
        log = '\r==>Eval: || Total_time: {time:.3f}s || loc_loss: {loc_loss:.4f} conf_loss: {conf_loss:.4f} || mAP: {mAP:.6f}\n'.format(mAP=ap,
                time=_t.total_time, loc_loss=loc_loss/epoch_size, conf_loss=conf_loss/epoch_size)
        sys.stdout.write(log)
        sys.stdout.flush()

        # log for tensorboard
        writer.add_scalar('Eval/loc_loss', loc_loss/epoch_size, epoch)
        writer.add_scalar('Eval/conf_loss', conf_loss/epoch_size, epoch)
        writer.add_scalar('Eval/mAP', ap, epoch)
        viz_pr_curve(writer, prec, rec, epoch)
        viz_archor_strategy(writer, size, gt_label, epoch)


    def test_epoch(self, model, data_loader, detector, output_dir, use_gpu, writer=None, epoch=None, printout=True):
        model.eval()

        # # use this line below if you wish to prune the model. Check prune.py for details
        model = prune_model(model, 0.8)

        dataset = data_loader.dataset
        num_images = len(dataset)
        num_classes = detector.num_classes
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        empty_array = np.transpose(np.array([[],[],[],[],[]]),(1,0))

        _t = Timer()
        _t_model = Timer()
        fps_list = [] #to track fps for calculation of average fps
        fps_model = [] #to track fps accounting only for model's forward time
        mAP_list = [] #to track mAP for each image
        for i in iter(range((num_images))):
            img = dataset.pull_image(i)
            scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            if use_gpu:
                images = Variable(dataset.preproc(img)[0].unsqueeze(0).cuda(), volatile=True)
            else:
                images = Variable(dataset.preproc(img)[0].unsqueeze(0), volatile=True)

            #resets the LSTM at specified points to treat next frame as new video
            if self.test_video_break:
                if i in self.test_video_break:
                    reset_model_LSTM(model)

            _t.tic()
            _t_model.tic()
            out = model(images, phase='eval')
            time_model = _t_model.toc()
            detections = detector.forward(out)

            time = _t.toc()

            if self.cfg.TEST.PRINT_IMAGES: print_detections(images, detections)

            intermittent_box = [[[]] for _ in range(num_classes)] #for intermittent measuring of mAP
            # TODO: make it smart:
            for j in range(1, num_classes):
                cls_dets = list()
                for det in detections[0][j]:
                    if det[0] > 0:
                        d = det.cpu().numpy()
                        score, box = d[0], d[1:]
                        box *= scale
                        box = np.append(box, score)
                        cls_dets.append(box)
                if len(cls_dets) == 0:
                    cls_dets = empty_array
                all_boxes[j][i] = np.array(cls_dets)
                intermittent_box[j] = np.array(cls_dets) #for intermittent measuring of mAP
                
            fps_list.append(1/time)
            fps_model.append(1/time_model)
            # if self.cfg.DATASET.DATASET == 'customRNN':
            #     #intermittent evaluation of mAP. 
            #     #There still seems to be some difference between this and evaluating at the end
            #     APs, mAP = data_loader.dataset.evaluate_intermittent_detections(intermittent_box, img_id=i, output_dir=output_dir)
            #     mAP_list.append(mAP)
            #     writer.add_scalar('mAP/image_num', mAP, i)
            # # log per iter
            #     log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}] mAP:{mAP:.3f}\r'.format(
            #             prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
            #             time=time, mAP=mAP)
            # else:
            log = '\r==>Test: || {iters:d}/{epoch_size:d} in {time:.3f}s [{prograss}]'.format(
                    prograss='#'*int(round(10*i/num_images)) + '-'*int(round(10*(1-i/num_images))), iters=i, epoch_size=num_images,
                    time=time)
             
            sys.stdout.write(log)
            sys.stdout.flush()
        print('Average fps: {fps:.3f}'.format(fps=np.mean(np.array(fps_list)))) #print the average fps for this epoch
        print('Average model fps without accounting for processing of output: {fps:.3f}'.format(fps=np.mean(np.array(fps_model)))) #print the average fps for this epoch
        
        # write result to pkl
        with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        # currently the COCO dataset do not return the mean ap or ap 0.5:0.95 values
        print('Evaluating detections')
        _, mAP = data_loader.dataset.evaluate_detections(all_boxes, output_dir)
        if epoch is not None:
            print('mAP: {mAP:.3f}'.format(mAP=mAP))
            writer.add_scalar('Train/test_mAP', mAP, epoch)

    def visualize_epoch(self, model, data_loader, priorbox, writer, epoch, use_gpu):
        model.eval()

        img_index = random.randint(0, len(data_loader.dataset)-1)

        # get img
        image = data_loader.dataset.pull_image(img_index)
        anno = data_loader.dataset.pull_anno(img_index)

        # visualize archor box
        viz_prior_box(writer, priorbox, image, epoch)

        # get preproc
        preproc = data_loader.dataset.preproc
        preproc.add_writer(writer, epoch)
        # preproc.p = 0.6

        # preproc image & visualize preprocess prograss
        images = Variable(preproc(image, anno)[0].unsqueeze(0), volatile=True)
        if use_gpu:
            images = images.cuda()

        # visualize feature map in base and extras
        base_out = viz_module_feature_maps(writer, model.base, images, module_name='base', epoch=epoch)
        extras_out = viz_module_feature_maps(writer, model.extras, base_out, module_name='extras', epoch=epoch)
        # visualize feature map in feature_extractors
        viz_feature_maps(writer, model(images, 'feature'), module_name='feature_extractors', epoch=epoch)

        model.train()
        images.requires_grad = True
        images.volatile=False
        base_out = viz_module_grads(writer, model, model.base, images, images, preproc.means, module_name='base', epoch=epoch)

        # TODO: add more...


    def configure_optimizer(self, trainable_param, cfg):
        if cfg.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'rmsprop':
            optimizer = optim.RMSprop(trainable_param, lr=cfg.LEARNING_RATE,
                        momentum=cfg.MOMENTUM, alpha=cfg.MOMENTUM_2, eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        elif cfg.OPTIMIZER == 'adam':
            optimizer = optim.Adam(trainable_param, lr=cfg.LEARNING_RATE,
                        betas=(cfg.MOMENTUM, cfg.MOMENTUM_2), eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
        else:
            AssertionError('optimizer can not be recognized.')
        return optimizer


    def configure_lr_scheduler(self, optimizer, cfg):
        if cfg.SCHEDULER == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEPS[0], gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'multi_step':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.STEPS, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.GAMMA)
        elif cfg.SCHEDULER == 'SGDR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.MAX_EPOCHS)
        else:
            AssertionError('scheduler can not be recognized.')
        return scheduler


    def export_graph(self):
        self.model.train(False)
        dummy_input = Variable(torch.randn(1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])).cuda()
        # Export the model
        torch_out = torch.onnx._export(self.model,             # model being run
                                       dummy_input,            # model input (or a tuple for multiple inputs)
                                       "graph.onnx",           # where to save the model (can be a file or file-like object)
                                       export_params=True)     # store the trained parameter weights inside the model file
        # if not os.path.exists(cfg.EXP_DIR):
        #     os.makedirs(cfg.EXP_DIR)
        # self.writer.add_graph(self.model, (dummy_input, ))


def train_model():
    s = Solver()
    s.train_model()
    return True

def test_model():
    s = Solver()
    s.test_model()
    return True
