import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os
from packaging import version

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from model.deeplab import Res_Deeplab
from dataset.voc_dataset import VOCDataSet
from model.deeplab import Res_Deeplab
from PIL import Image

import matplotlib.pyplot as plt

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set.
#RESTORE_FROM = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-8d75b3f1.pth'
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

pretrianed_models_dict ={'semi0.125': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-03c6f81c.pth',
                         'semi0.25': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.25-473f8a14.pth',
                         'semi0.5': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.5-acf6a654.pth',
                         'advFull': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSegVOCFull-92fbc7ee.pth'}


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default='',
                        help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))


    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')

def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5),
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5),
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0),
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    model = Res_Deeplab(num_classes=args.num_classes)

    # if args.pretrained_model != None:
    #     args.restore_from = pretrianed_models_dict[args.pretrained_model]
    #
    # if args.restore_from[:4] == 'http' :
    #     saved_state_dict = model_zoo.load_url(args.restore_from)
    # else:
    #     saved_state_dict = torch.load(args.restore_from)
    #model.load_state_dict(saved_state_dict)

    model = Res_Deeplab(num_classes=args.num_classes)
    #model.load_state_dict(torch.load('/data/wyc/AdvSemiSeg/snapshots/VOC_15000.pth'))
    state_dict=torch.load('/data1/wyc/AdvSemiSeg/snapshots/VOC_t_concat_pred_img_15000.pth')
    from model.discriminator_pred_concat_img import FCDiscriminator

    model_D = FCDiscriminator(num_classes=args.num_classes)

    state_dict_d = torch.load('/data1/wyc/AdvSemiSeg/snapshots/VOC_t_concat_pred_img_15000_D.pth')


    # original saved file with DataParallel

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        print (name)
        if name in new_state_dict and param.size() == new_state_dict[name].size():
            new_params[name].copy_(new_state_dict[name])
            print('copy {}'.format(name))

    model.load_state_dict(new_params)

    model.eval()
    model.cuda()

    new_state_dict_d = OrderedDict()
    for k, v in state_dict_d.items():
        name = k[7:]  # remove `module.`
        new_state_dict_d[name] = v

    new_params_d = model_D.state_dict().copy()
    for name, param in new_params_d.items():

        print (name)
        if name in new_state_dict_d and param.size() == new_state_dict_d[name].size():
            new_params_d[name].copy_(new_state_dict_d[name])
            print('copy {}'.format(name))

    model_D.load_state_dict(new_params_d)

    model_D.eval()
    model_D.cuda()

    testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=True)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(505, 505), mode='bilinear')
    data_list = []

    colorize = VOCColorize()

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd'%(index))
        image, label, size, name = batch
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda())
        image_d=Variable(image, volatile=True).cuda()
        output=interp(output)
        output_dout = output.clone()
        output_pred = F.softmax(output, dim=1).cpu().data[0].numpy()
        label2=label[0].numpy()
        output = output.cpu().data[0].numpy()
        output = output[:,:size[0],:size[1]]
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        #"""pred result"""
        filename = os.path.join(args.save_dir, '{}.png'.format(name[0]))
        color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
        color_file.save(filename)


        #"""the area of the pred which is wrong"""
        output_mistake=np.zeros(output.shape)
        semi_ignore_mask_correct = (output == gt)
        #semi_ignore_mask_255=(gt==255)
        output_mistake[semi_ignore_mask_correct] = 255
        #output_mistake[semi_ignore_mask_255] = 255
        output_mistake = np.expand_dims(output_mistake, axis=2)
        filename2 = os.path.join('/data1/wyc/AdvSemiSeg/pred_mis/', '{}.png'.format(name[0]))
        cv2.imwrite(filename2, output_mistake)



        #"""dis confidence map decide line of pred map"""
        D_out = interp(model_D(torch.cat([F.softmax(output_dout, dim=1),F.sigmoid(image_d)],1)))#67
        D_out_sigmoid = (F.sigmoid(D_out).data[0].cpu().numpy())
        D_out_sigmoid = D_out_sigmoid[:, :size[0], :size[1]]
        semi_ignore_mask_dout0 = (D_out_sigmoid < 0.0001)
        semi_ignore_mask_dout255 = (D_out_sigmoid >= 0.0001)

        D_out_sigmoid[semi_ignore_mask_dout0] = 0
        D_out_sigmoid[semi_ignore_mask_dout255] = 255
        filename2 = os.path.join('/data1/wyc/AdvSemiSeg/confidence_line/', '{}.png'.format(name[0]))#0 black 255 white
        cv2.imwrite(filename2,D_out_sigmoid.transpose(1, 2, 0))


        #""""pred max decide line of pred map"""
        # id2 = np.argmax(output_pred, axis=0)
        # map=np.zeros([1,id2.shape[0],id2.shape[1]])
        # for i in range(id2.shape[0]):
        #     for j in range(id2.shape[1]):
        #         map[0][i][j]=output_pred[id2[i][j]][i][j]
        # semi_ignore_mask2 = (map < 0.999999)
        # semi_ignore_mask3 = (map >= 0.999999)
        # map[semi_ignore_mask2] = 0
        # map[semi_ignore_mask3] = 255
        # map = map[:, :size[0], :size[1]]
        # filename2 = os.path.join('/data1/wyc/AdvSemiSeg/pred_line/', '{}.png'.format(name[0]))#0 black 255 white
        # cv2.imwrite(filename2,map.transpose(1, 2, 0))





        data_list.append([gt.flatten(), output.flatten()])

    filename = os.path.join(args.save_dir, 'result.txt')
    get_iou(data_list, args.num_classes, filename)


if __name__ == '__main__':
    main()
