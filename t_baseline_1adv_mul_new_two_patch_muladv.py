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
from model.my_discriminator import Discriminator2,Discriminator2_mul
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
                --lambda-semi 0.1 --semi-start 5000 --mask-T =fv0.2
"""
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
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
    tag = 0

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


    model.train()
    model=nn.DataParallel(model)
    model.cuda()


    cudnn.benchmark = True

    # init D
    model_DS = []
    for i in range(20):
        model_DS.append(Discriminator2_mul(num_classes=args.num_classes))
    # if args.restore_from_D is not None:
    #      model_D.load_state_dict(torch.load(args.restore_from_D))
    for model_D in model_DS:
        model_D = nn.DataParallel(model_D)
        model_D.train()
        model_D.cuda()

    model_D2 = Discriminator2(num_classes=args.num_classes)
    if args.restore_from_D is not None:
        model_D2.load_state_dict(torch.load(args.restore_from_D))
    model_D2 = nn.DataParallel(model_D2)
    model_D2.train()
    model_D2.cuda()

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
    optimizer_DS = []
    for model_D in model_DS:
        optimizer_DS.append(optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99)))
    for optimizer_D in optimizer_DS:
        optimizer_D.zero_grad()


    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

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

        for optimizer_D in optimizer_DS:
            optimizer_D.zero_grad()
            adjust_learning_rate_D(optimizer_D, i_iter)

        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for model_D in model_DS:
                for param in model_D.parameters():
                    param.requires_grad = False
            for param in model_D2.parameters():
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


            pred_re0 = F.softmax(pred, dim=1)
            pred_re=pred_re0.repeat(1, 3, 1, 1)
            indices_1 = torch.index_select(images, 1, Variable(torch.LongTensor([0])).cuda())
            indices_2 = torch.index_select(images, 1, Variable(torch.LongTensor([1])).cuda())
            indices_3 = torch.index_select(images, 1, Variable(torch.LongTensor([2])).cuda())
            img_re = torch.cat(
                [indices_1.repeat(1, 21, 1, 1), indices_2.repeat(1, 21, 1, 1), indices_3.repeat(1, 21, 1, 1), ], 1)

            mul_img = pred_re * img_re#10,63,321,321

            D_out_2 = model_D2(mul_img)

            loss_adv_pred_ = 0

            for i_l in range(labels.shape[0]):
                label_set=np.unique(labels[i_l]).tolist()
                for ls in label_set:
                    if ls !=0 and ls!=255:
                        ls=int(ls)

                        img_p = torch.cat(
                            [mul_img[i_l][ls].unsqueeze(0).unsqueeze(0), mul_img[i_l][ls+ 21].unsqueeze(0).
                            unsqueeze(0), mul_img[i_l][ls+ 21 + 21].unsqueeze(0).unsqueeze(0)], 1)
                        #print 1,img_p.size()#(1L, 3L, 321L, 321L)
                        imgs=img_p
                        imgs1 = imgs[:,:,0:107, 0:107]
                        imgs2 = imgs[:,:,0:107, 107:214]
                        imgs3 = imgs[:,:,0:107, 214:321]
                        imgs4 = imgs[:,:,107:214, 0:107]
                        imgs5 = imgs[:,:,107:214, 107:214]
                        imgs6 = imgs[:,:,107:214, 214:321]
                        imgs7 = imgs[:,:,214:321, 0:107]
                        imgs8 = imgs[:,:,214:321, 107:214]
                        imgs9 = imgs[:,:,214:321, 214:321]

                        #print 2, imgs1.size()#(1L, 3L, 107L, 107L)

                        img_ps=torch.cat([imgs1,imgs2,imgs3,imgs4,imgs5,imgs6,imgs7,imgs8,imgs9],0)
                        #print 3, img_ps.size()#(9L, 3L, 107L, 107L)

                        D_out = model_DS[ls-1](img_ps)
                        loss_adv_pred_=loss_adv_pred_+bce_loss(D_out, make_D_label(gt_label, D_out))


            loss_adv_pred = loss_adv_pred_*0.5+bce_loss(D_out_2, make_D_label(gt_label,D_out_2))

            loss = loss_seg + args.lambda_adv_pred * loss_adv_pred

            # proper normalization
            loss = loss/args.iter_size
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy()[0]/args.iter_size
            loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()[0]/args.iter_size


            # train D

            # bring back requires_grad
            for model_D in model_DS:
                for param in model_D.parameters():
                    param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with pred
            pred = pred.detach()

            pred_re0 = F.softmax(pred, dim=1)
            pred_re2 = pred_re0.repeat(1, 3, 1, 1)

            mul_img2 = pred_re2 * img_re


            D_out_2 = model_D2(mul_img2)

            loss_adv_pred_ = 0

            for i_l in range(labels.shape[0]):
                label_set = np.unique(labels[i_l]).tolist()
                for ls in label_set:
                    if ls != 0 and ls != 255:
                        ls = int(ls)

                        img_p = torch.cat(
                            [mul_img2[i_l][ls].unsqueeze(0).unsqueeze(0), mul_img2[i_l][ls+ 21].unsqueeze(0).
                                unsqueeze(0), mul_img2[i_l][ls + 21 + 21].unsqueeze(0).unsqueeze(0)], 1)
                        # print 1,img_p.size()#(1L, 3L, 321L, 321L)
                        imgs = img_p
                        imgs1 = imgs[:, :, 0:107, 0:107]
                        imgs2 = imgs[:, :, 0:107, 107:214]
                        imgs3 = imgs[:, :, 0:107, 214:321]
                        imgs4 = imgs[:, :, 107:214, 0:107]
                        imgs5 = imgs[:, :, 107:214, 107:214]
                        imgs6 = imgs[:, :, 107:214, 214:321]
                        imgs7 = imgs[:, :, 214:321, 0:107]
                        imgs8 = imgs[:, :, 214:321, 107:214]
                        imgs9 = imgs[:, :, 214:321, 214:321]

                        # print 2, imgs1.size()#(1L, 3L, 107L, 107L)

                        img_ps = torch.cat([imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, imgs7, imgs8, imgs9], 0)
                        # print 3, img_ps.size()#(9L, 3L, 107L, 107L)

                        D_out = model_DS[ls - 1](img_ps)
                        loss_adv_pred_ = loss_adv_pred_ + bce_loss(D_out, make_D_label(pred_label, D_out))



            loss_D = loss_adv_pred_*0.5+bce_loss(D_out_2, make_D_label(pred_label,D_out_2))
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

            img_gt, labels_gt, _, name = batch
            img_gt=Variable(img_gt).cuda()
            D_gt_v = Variable(one_hot(labels_gt)).cuda()
            ignore_mask_gt = (labels_gt.numpy() == 255)

            lb=D_gt_v.detach()

            pred_re3 = D_gt_v.repeat(1, 3, 1, 1)
            indices_1 = torch.index_select(img_gt, 1, Variable(torch.LongTensor([0])).cuda())
            indices_2 = torch.index_select(img_gt, 1, Variable(torch.LongTensor([1])).cuda())
            indices_3 = torch.index_select(img_gt, 1, Variable(torch.LongTensor([2])).cuda())
            img_re3 = torch.cat(
                [indices_1.repeat(1, 21, 1, 1), indices_2.repeat(1, 21, 1, 1), indices_3.repeat(1, 21, 1, 1), ], 1)

            #mul_img3 = img_re3
            mul_img3 = pred_re3 * img_re3


            D_out_2 = model_D2(mul_img3)

            loss_adv_pred_ = 0

            for i_l in range(labels_gt.shape[0]):
                label_set = np.unique(labels_gt[i_l]).tolist()

                for ls in label_set:
                    if ls != 0 and ls != 255:
                        ls = int(ls)

                        img_p = torch.cat(
                            [mul_img3[i_l][ls].unsqueeze(0).unsqueeze(0), mul_img3[i_l][ls+ 21].unsqueeze(0).
                                unsqueeze(0), mul_img3[i_l][ls+ 21 + 21].unsqueeze(0).unsqueeze(0)], 1)
                        # print 1,img_p.size()#(1L, 3L, 321L, 321L)
                        imgs = img_p
                        imgs1 = imgs[:, :, 0:107, 0:107]
                        imgs2 = imgs[:, :, 0:107, 107:214]
                        imgs3 = imgs[:, :, 0:107, 214:321]
                        imgs4 = imgs[:, :, 107:214, 0:107]
                        imgs5 = imgs[:, :, 107:214, 107:214]
                        imgs6 = imgs[:, :, 107:214, 214:321]
                        imgs7 = imgs[:, :, 214:321, 0:107]
                        imgs8 = imgs[:, :, 214:321, 107:214]
                        imgs9 = imgs[:, :, 214:321, 214:321]

                        # print 2, imgs1.size()#(1L, 3L, 107L, 107L)

                        img_ps = torch.cat([imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, imgs7, imgs8, imgs9], 0)
                        # print 3, img_ps.size()#(9L, 3L, 107L, 107L)

                        D_out = model_DS[ls - 1](img_ps)
                        loss_adv_pred_ = loss_adv_pred_ + bce_loss(D_out, make_D_label(gt_label, D_out))


                        '''


                        if tag == 0:
                            # print lb[0].size()
                            # lb1=lb[0][0]
                            # lb2 = lb[0][1]
                            # lb3 = lb[0][2]
                            # lb4 = lb[0][3]
                            # lb5 = lb[0][4]
                            # lb6 = lb[0][5]
                            # lb7 = lb[0][0]
                            # lb8 = lb[0][0]
                            # lb9 = lb[0][0]
                            # lb10 = lb[0][0]


                            print label_set, name[0]
                            print ls
                            imgs = imgs.squeeze()
                            imgs = imgs.transpose(0, 1)
                            imgs = imgs.transpose(1, 2)

                            imgs1 = imgs1.squeeze()
                            imgs1 = imgs1.transpose(0, 1)
                            imgs1 = imgs1.transpose(1, 2)

                            imgs2 = imgs2.squeeze()
                            imgs2 = imgs2.transpose(0, 1)
                            imgs2 = imgs2.transpose(1, 2)

                            imgs3 = imgs3.squeeze()
                            imgs3 = imgs3.transpose(0, 1)
                            imgs3 = imgs3.transpose(1, 2)

                            imgs4 = imgs4.squeeze()
                            imgs4 = imgs4.transpose(0, 1)
                            imgs4 = imgs4.transpose(1, 2)

                            imgs5 = imgs5.squeeze()
                            imgs5 = imgs5.transpose(0, 1)
                            imgs5 = imgs5.transpose(1, 2)

                            imgs6 = imgs6.squeeze()
                            imgs6 = imgs6.transpose(0, 1)
                            imgs6 = imgs6.transpose(1, 2)

                            imgs7 = imgs7.squeeze()
                            imgs7 = imgs7.transpose(0, 1)
                            imgs7 = imgs7.transpose(1, 2)

                            imgs8 = imgs8.squeeze()
                            imgs8 = imgs8.transpose(0, 1)
                            imgs8 = imgs8.transpose(1, 2)

                            imgs9 = imgs9.squeeze()
                            imgs9 = imgs9.transpose(0, 1)
                            imgs9 = imgs9.transpose(1, 2)

                            imgs = imgs.data.cpu().numpy()
                            imgs1 = imgs1.data.cpu().numpy()
                            imgs2 = imgs2.data.cpu().numpy()
                            imgs3 = imgs3.data.cpu().numpy()
                            imgs4 = imgs4.data.cpu().numpy()
                            imgs5 = imgs5.data.cpu().numpy()
                            imgs6 = imgs6.data.cpu().numpy()
                            imgs7 = imgs7.data.cpu().numpy()
                            imgs8 = imgs8.data.cpu().numpy()
                            imgs9 = imgs9.data.cpu().numpy()
                            cv2.imwrite('/data1/wyc/1.png', imgs1)
                            cv2.imwrite('/data1/wyc/2.png', imgs2)
                            cv2.imwrite('/data1/wyc/3.png', imgs3)
                            cv2.imwrite('/data1/wyc/4.png', imgs4)
                            cv2.imwrite('/data1/wyc/5.png', imgs5)
                            cv2.imwrite('/data1/wyc/6.png', imgs6)
                            cv2.imwrite('/data1/wyc/7.png', imgs7)
                            cv2.imwrite('/data1/wyc/8.png', imgs8)
                            cv2.imwrite('/data1/wyc/9.png', imgs9)
                            cv2.imwrite('/data1/wyc/img.png', imgs)
                            tag = 1
                        '''


            loss_D = loss_adv_pred_*0.5+bce_loss(D_out_2, make_D_label(gt_label,D_out_2))
            loss_D = loss_D/args.iter_size/2
            loss_D.backward()
            loss_D_value += loss_D.data.cpu().numpy()[0]



        optimizer.step()

        for optimizer_D in optimizer_DS:
            optimizer_D.step()

        optimizer_D2.step()

        print('exp = {}'.format(args.snapshot_dir))
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
