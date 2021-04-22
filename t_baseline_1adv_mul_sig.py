import argparse
import cv2
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
#import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
import pickle
from packaging import version

from model.my_deeplab import Res_Deeplab
from model.my_discriminator import Discriminator2
from utils.my_loss import CrossEntropy2d, BCEWithLogitsLoss2d
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet
import math


import matplotlib.pyplot as plt
import random
import timeit
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 10
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/train_aug.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 21
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1

PARTIAL_DATA=0.5

SEMI_START=5000
LAMBDA_SEMI=0.1
MASK_T=0.2

LAMBDA_SEMI_ADV=0.001
SEMI_START_ADV=0
D_REMAIN=True


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--partial-data", type=float, default=PARTIAL_DATA,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--partial-id", type=str, default=None,
                        help="restore partial id list")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-pred", type=float, default=LAMBDA_ADV_PRED,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-semi", type=float, default=LAMBDA_SEMI,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--lambda-semi-adv", type=float, default=LAMBDA_SEMI_ADV,
                        help="lambda_semi for adversarial training.")
    parser.add_argument("--mask-T", type=float, default=MASK_T,
                        help="mask T for semi adversarial training.")
    parser.add_argument("--semi-start", type=int, default=SEMI_START,
                        help="start semi learning after # iterations")
    parser.add_argument("--semi-start-adv", type=int, default=SEMI_START_ADV,
                        help="start semi learning after # iterations")
    parser.add_argument("--D-remain", type=bool, default=D_REMAIN,
                        help="Whether to train D with unlabeled data")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-D", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default="0",
                        help="choose gpu device.")
    return parser.parse_args()
"""
--snapshot-dir snapshots \
                --partial-data 0.125 \
                --num-steps 20000 \
                --lambda-adv-pred 0.01 \
                --lambda-semi 0.1 --semi-start 5000 --mask-T 0.2
"""
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
args = get_arguments()

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10

def one_hot(label):
    label = label.numpy()
    one_hot = np.zeros((label.shape[0], args.num_classes, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(args.num_classes):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot)



def make_D_label(label, D_out):
    D_label = np.ones(D_out.size())*label
    D_label = Variable(torch.FloatTensor(D_label)).cuda()
    return D_label


def main():

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True


    # create network
    model = Res_Deeplab(num_classes=args.num_classes)

    # load pretrained parameters
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

        # only copy the params that exist in current model (caffe-like)
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print (name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            print('copy {}'.format(name))
    model.load_state_dict(new_params)



    # only copy the params that exist in current model (caffe-like)

    # state_dict = torch.load(
    #     '/data1/wyc/AdvSemiSeg/snapshots/VOC_t_baseline_1adv_mul_20000.pth')  # baseline707 adv 709 nadv 705()*2
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    #
    # new_params = model.state_dict().copy()
    # for name, param in new_params.items():
    #     print (name)
    #     if name in new_state_dict and param.size() == new_state_dict[name].size():
    #         new_params[name].copy_(new_state_dict[name])
    #         print('copy {}'.format(name))
    #
    # model.load_state_dict(new_params)


    model.train()
    model=nn.DataParallel(model)
    model.cuda()


    cudnn.benchmark = True

    # init D
    model_D = Discriminator2(num_classes=args.num_classes)
    if args.restore_from_D is not None:
         model_D.load_state_dict(torch.load(args.restore_from_D))
    model_D = nn.DataParallel(model_D)
    model_D.train()
    model_D.cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


    train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    train_dataset_size = len(train_dataset)

    train_gt_dataset = VOCGTDataSet(args.data_dir, args.data_list, crop_size=input_size,
                       scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

    if args.partial_data ==0:
        trainloader = data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)

        trainloader_gt = data.DataLoader(train_gt_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    else:
        #sample partial data
        partial_size = int(args.partial_data * train_dataset_size)



    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)


    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.SGD(model.module.optim_parameters(args),
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9,0.99))
    optimizer_D.zero_grad()

    # loss/ bilinear upsampling
    bce_loss = torch.nn.BCELoss()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')


    # labels for adversarial training
    pred_label = 0
    gt_label = 1


    for i_iter in range(args.num_steps):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_D_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        tw=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source
            try:
                _, batch = trainloader_iter.next()
            except:
                trainloader_iter = enumerate(trainloader)
                _, batch = trainloader_iter.next()

            images, labels, _, _ = batch
            images = Variable(images).cuda()
            ignore_mask = (labels.numpy() == 255)
            pred = interp(model(images))
            loss_seg = loss_calc(pred, labels)

            pred_0 = F.softmax(pred, dim=1)
            #pred_0 = 1 / (math.e ** (((pred_01 - 0.33) * 30) * (-1)) + 1)



            labels0 = Variable(one_hot(labels)).cuda()
            one_s = Variable(torch.ones(labels0.size())).cuda()
            labels0=one_s-labels0
            labels0 = torch.index_select(labels0, 1, Variable(
                torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])).cuda())
            pred0 = torch.index_select(pred_0, 1, Variable(
                torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])).cuda())

            pred_label0 = labels0 * pred0
            pred_max = torch.max(pred_0, dim=1)[1]

            pred_max = Variable(one_hot(pred_max.cpu().data)).cuda()

            pred_max = torch.index_select(pred_max, 1, Variable(
                torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])).cuda())

            pred_c = pred_label0 * (pred_max)
            one_s = Variable(torch.ones(pred_max.size())).cuda()
            pred_m = pred_label0 * (one_s - pred_max)

            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0
            c8 = 0
            c9 = 0
            c10 = 0
            c0 = 0

            pred_min_c_list = pred_c.cpu().data.numpy().flatten().tolist()
            pred_min_c_l = len(pred_min_c_list)

            for n in pred_min_c_list:
                if n < 0.00000001:
                    c0 = c0 + 1
                elif n < 0.1:
                    c3 = c3 + 1
                elif n < 0.2:
                    c4 = c4 + 1
                elif n < 0.3:
                    c5 = c5 + 1
                elif n < 0.4:
                    c6 = c6 + 1
                elif n < 0.5:
                    c7 = c7 + 1
                elif n < 0.6:
                    c8 = c8 + 1
                elif n < 0.7:
                    c9 = c9 + 1
                elif n <=1:
                    c10 = c10 + 1
                else:
                    print n

            if pred_min_c_l - c0 == 0:
                print pred_min_c_l
            else:
                pred_min_c_l = (pred_min_c_l - c0) * 1.00000
                print "correct", 3, ":", c3 / pred_min_c_l, 4, ":", c4 / pred_min_c_l, 5, ":", c5 / pred_min_c_l, 6, ":", c6 / pred_min_c_l, 7, ":", c7 / pred_min_c_l, 8, ":", c8 / pred_min_c_l, 9, ":", c9 / pred_min_c_l, 10, ":", c10 / pred_min_c_l

            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0
            c8 = 0
            c9 = 0
            c10 = 0
            c0 = 0

            pred_min_m_list = pred_m.cpu().data.numpy().flatten().tolist()
            pred_min_c_l = len(pred_min_m_list)

            for n in pred_min_m_list:
                if n < 0.0000001:
                    c0 = c0 + 1
                elif n < 0.1:
                    c3 = c3 + 1
                elif n < 0.2:
                    c4 = c4 + 1
                elif n < 0.3:
                    c5 = c5 + 1
                elif n < 0.4:
                    c6 = c6 + 1
                elif n < 0.5:
                    c7 = c7 + 1
                else:
                    print n
            if pred_min_c_l - c0 == 0:
                print pred_min_c_l
            else:
                pred_min_c_l = (pred_min_c_l - c0) * 1.00000
                print "mistake", 1, ":", c3 / pred_min_c_l, 2, ":", c4 / pred_min_c_l, 3, ":", c5 / pred_min_c_l, 4, ":", c6 / pred_min_c_l, 5, ":", c7 / pred_min_c_l

            '''

            images, labels, _, _ = batch
            images = Variable(images).cuda()
            ignore_mask = (labels.numpy() == 255)
            pred = interp(model(images))
            loss_seg = loss_calc(pred, labels)

            pred_0 = F.softmax(pred, dim=1)

            labels0 = Variable(one_hot(labels)).cuda()
            labels0=torch.index_select(labels0, 1, Variable(torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])).cuda())
            pred0 = torch.index_select(pred_0, 1, Variable(torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])).cuda())


            pred_label0=labels0*pred0
            pred_max = torch.max(pred_0, dim=1)[1]


            pred_max = Variable(one_hot(pred_max.cpu().data)).cuda()

            pred_max = torch.index_select(pred_max, 1, Variable(
                torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])).cuda())


            pred_c=pred_label0*(pred_max)
            one_s = Variable(torch.ones(pred_max.size())).cuda()
            pred_m = pred_label0 * (one_s - pred_max)

            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0
            c8 = 0
            c9 = 0
            c10 = 0
            c0 = 0

            pred_min_c_list = pred_c.cpu().data.numpy().flatten().tolist()
            pred_min_c_l = len(pred_min_c_list)

            for n in pred_min_c_list:
                if n < 0.00000001:
                    c0 = c0 + 1
                elif n < 0.3:
                    c3 = c3 + 1
                elif n < 0.4:
                    c4 = c4 + 1
                elif n < 0.5:
                    c5 = c5 + 1
                elif n < 0.6:
                    c6 = c6 + 1
                elif n < 0.7:
                    c7 = c7 + 1
                elif n < 0.8:
                    c8 = c8 + 1
                elif n < 0.9:
                    c9 = c9 + 1
                elif n <= 1:
                    c10 = c10 + 1
                else:
                    print n

            if pred_min_c_l - c0 == 0:
                print pred_min_c_l
            else:
                pred_min_c_l = (pred_min_c_l - c0) * 1.00000
                print "correct", 3, ":", c3 / pred_min_c_l, 4, ":", c4 / pred_min_c_l, 5, ":", c5 / pred_min_c_l, 6, ":", c6 / pred_min_c_l, 7, ":", c7 / pred_min_c_l, 8, ":", c8 / pred_min_c_l, 9, ":", c9 / pred_min_c_l, 10, ":", c10 / pred_min_c_l

            c3 = 0
            c4 = 0
            c5 = 0
            c6 = 0
            c7 = 0
            c8 = 0
            c9 = 0
            c10 = 0
            c0 = 0

            pred_min_m_list = pred_m.cpu().data.numpy().flatten().tolist()
            pred_min_c_l = len(pred_min_m_list)


            for n in pred_min_m_list:
                if n < 0.0000001:
                    c0 = c0 + 1
                elif n < 0.1:
                    c3 = c3 + 1
                elif n < 0.2:
                    c4 = c4 + 1
                elif n < 0.3:
                    c5 = c5 + 1
                elif n < 0.4:
                    c6 = c6 + 1
                elif n < 0.5:
                    c7 = c7 + 1
                else:
                    print n
            if pred_min_c_l - c0 == 0:
                print pred_min_c_l
            else:
                pred_min_c_l = (pred_min_c_l - c0) * 1.00000
                print "mistake", 1, ":", c3 / pred_min_c_l, 2, ":", c4 / pred_min_c_l, 3, ":", c5 / pred_min_c_l, 4, ":", c6 / pred_min_c_l, 5, ":", c7 / pred_min_c_l
            '''




            # c3=0
            # c4=0
            # c5=0
            # c6=0
            # c7=0
            # c8=0
            # c9=0
            # c10=0
            # c0=0
            #
            # pred_min_c_list=pred_c.cpu().data.numpy().flatten().tolist()
            #
            # pred_min_c=set(pred_c.cpu().data.numpy().flatten().tolist())
            # pred_min_c_l=len(pred_min_c_list)
            # if len(pred_min_c) >1:
            #     pred_min_c.discard(0.0)
            #     pred_min_c=list(pred_min_c)
            #     pred_min_c2 = min(pred_min_c)
            #     pred_max_c2 = max(pred_min_c)
            # else:
            #     pred_min_c2 = 999
            #     pred_max_c2 = 999
            # for n in pred_min_c_list:
            #     if n<0.00000001:
            #         c0=c0+1
            #     elif n<0.3:
            #         c3=c3+1
            #     elif n<0.4:
            #         c4=c4+1
            #     elif n<0.5:
            #         c5=c5+1
            #     elif n<0.6:
            #         c6=c6+1
            #     elif n<0.7:
            #         c7=c7+1
            #     elif n<0.8:
            #         c8=c8+1
            #     elif n<0.9:
            #         c9=c9+1
            #     elif n<=1:
            #         c10=c10+1
            #     else:
            #         print n
            #
            # if pred_min_c_l-c0==0:
            #     print pred_min_c_l
            # else:
            #     pred_min_c_l=(pred_min_c_l-c0)*1.00000
            #
            # #print c0 + c3 + c4 + c5 + c6 + c7+c8+c9+c10
            #
            #     print "correct",3,":",c3/pred_min_c_l,4,":",c4/pred_min_c_l,5,":",c5/pred_min_c_l,6,":",c6/pred_min_c_l,7,":",c7/pred_min_c_l,8,":",c8/pred_min_c_l,9,":",c9/pred_min_c_l,10,":",c10/pred_min_c_l
            #
            # c3 = 0
            # c4 = 0
            # c5 = 0
            # c6 = 0
            # c7 = 0
            # c8 = 0
            # c9 = 0
            # c10 = 0
            # c0 = 0
            #
            #
            #
            #
            # pred_min_m_list = pred_m.cpu().data.numpy().flatten().tolist()
            # pred_min_m = set(pred_m.cpu().data.numpy().flatten().tolist())
            # pred_min_c_l = len(pred_min_m_list)
            # if len(pred_min_m) >1:
            #     pred_min_m.discard(0.0)
            #     pred_min_m=list(pred_min_m)
            #     pred_min_m2 = min(pred_min_m)
            #     pred_max_m2 = max(pred_min_m)
            # else:
            #     pred_min_m2 = 999
            #     pred_max_m2 = 999
            #
            # for n in pred_min_m_list:
            #     if n<0.0000001:
            #         c0=c0+1
            #     elif n<0.1:
            #         c3=c3+1
            #     elif n<0.2:
            #         c4=c4+1
            #     elif n<0.3:
            #         c5=c5+1
            #     elif n<0.4:
            #         c6=c6+1
            #     elif n<0.5:
            #         c7=c7+1
            #     else:
            #         print n
            # if pred_min_c_l-c0==0:
            #     print pred_min_c_l
            # else:
            #     pred_min_c_l=(pred_min_c_l-c0)*1.00000
            #     print "mistake",1,":",c3/pred_min_c_l,2,":",c4/pred_min_c_l,3,":",c5/pred_min_c_l,4,":",c6/pred_min_c_l,5,":",c7/pred_min_c_l
            #
            #
            #
            # print('max c  {}  min c  {}  max m  {}  min m  {}'.format(
            #                         pred_max_c2,pred_min_c2, pred_max_m2,pred_min_m2))

            '''20000
correct 3 : 6.60318801918e-05 4 : 0.00186540061542 5 : 0.00659328323715 6 : 0.0196708971091 7 : 0.0234446190621 8 : 0.0315071116335 9 : 0.0565364958202 10 : 0.860316160642
mistake 1 : 0.326503117461 2 : 0.211204060532 3 : 0.199110192061 4 : 0.15803490303 5 : 0.105147726917
max c  1.0  min c  0.206159025431  max m  0.499729216099  min m  3.0866687678e-11
iter =        0/   20000, loss_seg = 0.164, loss_adv_p = 0.688, loss_D = 0.687, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 1.03566126972e-05 4 : 0.000749128318431 5 : 0.00340042116892 6 : 0.0126937549625 7 : 0.0161735768288 8 : 0.0261400904478 9 : 0.0492767632133 10 : 0.891555908448
mistake 1 : 0.398261661669 2 : 0.20677876039 3 : 0.167001935704 4 : 0.129388545185 5 : 0.0985690970509
max c  1.0  min c  0.295039653778  max m  0.499661713839  min m  1.87631424287e-10
iter =        1/   20000, loss_seg = 0.114, loss_adv_p = 0.621, loss_D = 0.666, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.000170320224732 4 : 0.00122630561807 5 : 0.00573184329631 6 : 0.0155854360311 7 : 0.0216533779042 8 : 0.0339187050213 9 : 0.0683937894433 10 : 0.853320222461
mistake 1 : 0.370818679971 2 : 0.185940950356 3 : 0.165003680379 4 : 0.165167252801 5 : 0.113069436493
max c  1.0  min c  0.205323472619  max m  0.499007672071  min m  2.63143761003e-07
iter =        2/   20000, loss_seg = 0.136, loss_adv_p = 6.856, loss_D = 2.909, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 5.57311529183e-05 4 : 0.00162151116348 5 : 0.00529976725609 6 : 0.0104801106131 7 : 0.0125395094066 8 : 0.0183382031746 9 : 0.032151567505 10 : 0.919513599728
mistake 1 : 0.599129542479 2 : 0.138565514313 3 : 0.114044089027 4 : 0.0929903400446 5 : 0.0552705141361
max c  1.0  min c  0.251725673676  max m  0.499345749617  min m  2.17372370104e-11
iter =        3/   20000, loss_seg = 0.173, loss_adv_p = 0.211, loss_D = 1.186, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0 4 : 0.000408403449911 5 : 0.00314207170335 6 : 0.00972922412127 7 : 0.0128537300848 8 : 0.0230989478122 9 : 0.0531056227933 10 : 0.897662000035
mistake 1 : 0.376961004034 2 : 0.197477108279 3 : 0.160338093104 4 : 0.144009732983 5 : 0.1212140616
max c  1.0  min c  0.303397655487  max m  0.49908259511  min m  2.31815082538e-13

            '''

            '''0000
            
mistake 1 : 1.0 2 : 0.0 3 : 0.0 4 : 0.0 5 : 0.0
iter =        0/   20000, loss_seg = 3.045, loss_adv_p = 0.677, loss_D = 0.689, loss_semi = 0.000, loss_semi_adv = 0.000
20608200
mistake 1 : 1.0 2 : 0.0 3 : 0.0 4 : 0.0 5 : 0.0
iter =        1/   20000, loss_seg = 2.311, loss_adv_p = 0.001, loss_D = 4.009, loss_semi = 0.000, loss_semi_adv = 0.000
20608200
mistake 1 : 1.0 2 : 0.0 3 : 0.0 4 : 0.0 5 : 0.0
iter =        2/   20000, loss_seg = 2.695, loss_adv_p = 1.304, loss_D = 0.996, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.215316600114 4 : 0.28849591177 5 : 0.159402928313 6 : 0.113015782468 7 : 0.102595550485 8 : 0.0921563034797 9 : 0.0290169233695 10 : 0.0
mistake 1 : 0.80735288413 2 : 0.095622890725 3 : 0.0764600062066 4 : 0.0203387447147 5 : 0.000225474223205
iter =        3/   20000, loss_seg = 2.103, loss_adv_p = 0.936, loss_D = 0.720, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0 4 : 0.0106591946386 5 : 0.0159743486048 6 : 0.0144144664625 7 : 0.0482119128777 8 : 0.1030099948 9 : 0.149690912242 10 : 0.658039170374
mistake 1 : 0.997786787186 2 : 0.00220988967167 3 : 0.0 4 : 3.32314236342e-06 5 : 0.0
iter =        4/   20000, loss_seg = 3.231, loss_adv_p = 0.352, loss_D = 0.740, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0547200197126 4 : 0.0824408402489 5 : 0.105990337314 6 : 0.0824672410303 7 : 0.0818248220147 8 : 0.106500752422 9 : 0.12377566376 10 : 0.362280323498
mistake 1 : 0.89707796374 2 : 0.0776956426097 3 : 0.0212782533776 4 : 0.0038358600209 5 : 0.000112280251508
iter =        5/   20000, loss_seg = 2.440, loss_adv_p = 0.285, loss_D = 0.796, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0645113648539 4 : 0.11994706344 5 : 0.133270975533 6 : 0.130346143711 7 : 0.130461978634 8 : 0.155641595163 9 : 0.129387609717 10 : 0.136433268948
mistake 1 : 0.687583743536 2 : 0.172921294404 3 : 0.108947452597 4 : 0.027046718675 5 : 0.00350079078777
iter =        6/   20000, loss_seg = 1.562, loss_adv_p = 0.471, loss_D = 0.700, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0517260346307 4 : 0.0852126465864 5 : 0.0921211854525 6 : 0.0708675276672 7 : 0.0477657257266 8 : 0.0341796660139 9 : 0.0336846274009 10 : 0.584442586522
mistake 1 : 0.896840471376 2 : 0.0747503216715 3 : 0.0228403760663 4 : 0.00484719754372 5 : 0.000721633342184
iter =        7/   20000, loss_seg = 1.521, loss_adv_p = 1.028, loss_D = 0.781, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0354795814142 4 : 0.146293048623 5 : 0.182080980555 6 : 0.160111006186 7 : 0.0642910828885 8 : 0.0516294397657 9 : 0.0560234346393 10 : 0.304091425928
mistake 1 : 0.927571328438 2 : 0.0370632102681 3 : 0.0252651475979 4 : 0.00991933537868 5 : 0.000180978317074
iter =        8/   20000, loss_seg = 2.240, loss_adv_p = 1.128, loss_D = 1.030, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.013791534851 4 : 0.0218762276947 5 : 0.0420707290008 6 : 0.0509204695049 7 : 0.0551454624403 8 : 0.0738167607469 9 : 0.109429384722 10 : 0.632949431039
mistake 1 : 0.82698518928 2 : 0.0871808088324 3 : 0.056324796628 4 : 0.0222555690883 5 : 0.00725363617091
iter =        9/   20000, loss_seg = 1.620, loss_adv_p = 0.856, loss_D = 0.680, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0111066673332 4 : 0.0219300438268 5 : 0.039156626506 6 : 0.0714017897315 7 : 0.0741513772934 8 : 0.0834458164609 9 : 0.100755719975 10 : 0.598051958873
mistake 1 : 0.666796397198 2 : 0.104720338041 3 : 0.101412209496 4 : 0.0761078060714 5 : 0.0509632491938
iter =       10/   20000, loss_seg = 1.065, loss_adv_p = 0.474, loss_D = 0.706, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0158021057795 4 : 0.0165904591721 5 : 0.0267981756919 6 : 0.0441302710184 7 : 0.0592082596077 8 : 0.0964185397359 9 : 0.156169887236 10 : 0.584882301758
mistake 1 : 0.693582996679 2 : 0.111521356046 3 : 0.114913868543 4 : 0.0540284587444 5 : 0.0259533199871
iter =       11/   20000, loss_seg = 1.008, loss_adv_p = 0.442, loss_D = 0.724, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.00525433110547 4 : 0.0132143122692 5 : 0.0181547053526 6 : 0.0353758581661 7 : 0.0438108771263 8 : 0.057140850772 9 : 0.0862049023901 10 : 0.740844162818
mistake 1 : 0.698567424322 2 : 0.152478802192 3 : 0.0874738631406 4 : 0.0430223987244 5 : 0.0184575116206
iter =       12/   20000, loss_seg = 0.957, loss_adv_p = 0.442, loss_D = 0.743, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.00325696566801 4 : 0.0222856760435 5 : 0.029201151092 6 : 0.0513157992579 7 : 0.0617187558094 8 : 0.0804648983871 9 : 0.10184338308 10 : 0.649913370662
mistake 1 : 0.910824450383 2 : 0.0481368156486 3 : 0.0238175658895 4 : 0.0128053565929 5 : 0.00441581148608
iter =       13/   20000, loss_seg = 1.422, loss_adv_p = 0.830, loss_D = 0.618, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0149639294303 4 : 0.0338563367539 5 : 0.0443977524345 6 : 0.0406656984358 7 : 0.0549094069189 8 : 0.0846765553201 9 : 0.131008785505 10 : 0.595521535202
mistake 1 : 0.672923967008 2 : 0.177251807277 3 : 0.079961731589 4 : 0.0552797386879 5 : 0.0145827554388
iter =       14/   20000, loss_seg = 0.997, loss_adv_p = 1.154, loss_D = 0.837, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0193227170348 4 : 0.0525052357055 5 : 0.0709388706682 6 : 0.0720057691548 7 : 0.0699806377682 8 : 0.0911210337061 9 : 0.15581657249 10 : 0.468309163473
mistake 1 : 0.815034514788 2 : 0.1099266077 3 : 0.0481236762295 4 : 0.0204414796676 5 : 0.0064737216159
iter =       15/   20000, loss_seg = 0.938, loss_adv_p = 1.266, loss_D = 0.830, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.0138818759925 4 : 0.0214850753368 5 : 0.0245366000015 6 : 0.0524393902805 7 : 0.0717583953517 8 : 0.0829619547321 9 : 0.127629836154 10 : 0.605306872151
mistake 1 : 0.845186224931 2 : 0.0863806795239 3 : 0.037294979522 4 : 0.0189205299287 5 : 0.0122175860942
iter =       16/   20000, loss_seg = 1.872, loss_adv_p = 0.826, loss_D = 0.847, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.00228016623147 4 : 0.0089505718804 5 : 0.0161634364312 6 : 0.0323728439557 7 : 0.0323958295024 8 : 0.0427347284028 9 : 0.0699726012283 10 : 0.795129822368
mistake 1 : 0.815070892754 2 : 0.0785075426593 3 : 0.0488263539692 4 : 0.0359976506471 5 : 0.0215975599703
iter =       17/   20000, loss_seg = 1.717, loss_adv_p = 0.603, loss_D = 0.620, loss_semi = 0.000, loss_semi_adv = 0.000
correct 3 : 0.00200898622728 4 : 0.00781591308424 5 : 0.0269459263818 6 : 0.0502692998205 7 : 0.0554862862773 8 : 0.072763567832 9 : 0.10771673932 10 : 0.676993281057



            '''




            pred_re = F.softmax(pred, dim=1).repeat(1, 3, 1, 1)
            indices_1 = torch.index_select(images, 1, Variable(torch.LongTensor([0])).cuda())
            indices_2 = torch.index_select(images, 1, Variable(torch.LongTensor([1])).cuda())
            indices_3 = torch.index_select(images, 1, Variable(torch.LongTensor([2])).cuda())
            img_re = torch.cat(
                [indices_1.repeat(1, 21, 1, 1), indices_2.repeat(1, 21, 1, 1), indices_3.repeat(1, 21, 1, 1), ], 1)

            mul_img = pred_re * img_re


            D_out = model_D(mul_img)

            loss_adv_pred = bce_loss(D_out, make_D_label(gt_label,D_out))

            loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

            # proper normalization
            loss = loss/args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy()[0]/args.iter_size
            loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()[0]/args.iter_size


            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with pred
            pred = pred.detach()




            pred_re2 = F.softmax(pred, dim=1).repeat(1, 3, 1, 1)


            mul_img2 = pred_re2 * img_re

            D_out = model_D(mul_img2)

            loss_D = bce_loss(D_out, make_D_label(pred_label,D_out))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()[0]


            # train with gt
            # get gt labels
            try:
                _, batch = trainloader_gt_iter.next()
            except:
                trainloader_gt_iter = enumerate(trainloader_gt)
                _, batch = trainloader_gt_iter.next()

            img_gt, labels_gt, _, _ = batch
            img_gt=Variable(img_gt).cuda()
            D_gt_v = Variable(one_hot(labels_gt)).cuda()
            ignore_mask_gt = (labels_gt.numpy() == 255)

            pred_re3 = D_gt_v.repeat(1, 3, 1, 1)
            indices_1 = torch.index_select(img_gt, 1, Variable(torch.LongTensor([0])).cuda())
            indices_2 = torch.index_select(img_gt, 1, Variable(torch.LongTensor([1])).cuda())
            indices_3 = torch.index_select(img_gt, 1, Variable(torch.LongTensor([2])).cuda())
            img_re3 = torch.cat(
                [indices_1.repeat(1, 21, 1, 1), indices_2.repeat(1, 21, 1, 1), indices_3.repeat(1, 21, 1, 1), ], 1)

            mul_img3 = pred_re3 * img_re3


            D_out = model_D(mul_img3)

            loss_D = bce_loss(D_out, make_D_label(gt_label,D_out))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()[0]



        optimizer.step()
        optimizer_D.step()

        #print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_adv_p = {3:.3f}, loss_D = {4:.3f}, loss_semi = {5:.3f}, loss_semi_adv = {6:.3f}'.format(i_iter, args.num_steps, loss_seg_value, loss_adv_pred_value, loss_D_value, loss_semi_value, loss_semi_adv_value))

        if i_iter >= args.num_steps-1:
            print( 'save model ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+os.path.abspath(__file__).split('/')[-1].split('.')[0]+'_'+str(args.num_steps)+'.pth'))
            #torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+os.path.abspath(__file__).split('/')[-1].split('.')[0]+'_'+str(args.num_steps)+'_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter!=0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+os.path.abspath(__file__).split('/')[-1].split('.')[0]+'_'+str(i_iter)+'.pth'))
            #torch.save(model_D.state_dict(),osp.join(args.snapshot_dir, 'VOC_'+os.path.abspath(__file__).split('/')[-1].split('.')[0]+'_'+str(i_iter)+'_D.pth'))

    end = timeit.default_timer()
    print(end-start,'seconds')

if __name__ == '__main__':
    main()
