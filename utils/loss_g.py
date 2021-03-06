import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)



        # predict2=predict
        # target2=target
        #
        # n, c, h, w = predict2.size()
        # target_mask2 = (target2 >= 0) * (target2 != self.ignore_label)
        # target2= target2[target_mask2]
        # if not target2.data.dim():
        #     return Variable(torch.zeros(1))
        # predict2 = predict2.transpose(1, 2).transpose(2, 3).contiguous()
        # predict2 = predict2[target_mask2.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #
        #
        # D_out1=D_out.copy()
        # D_out1=Variable(torch.FloatTensor(D_out1)).cuda()
        # D_out3=((D_out1.view(n, h, w, 1).repeat(1, 1, 1, c))[target_mask2.view(n, h, w, 1).repeat(1, 1, 1, c)]).view(-1, c)
        # log_s=F.log_softmax(predict2)
        # logs2=log_s*D_out3
        # loss2=F.nll_loss(logs2, target2, weight, size_average=self.size_average, ignore_index=-100)
        return loss

class CrossEntropy2d2(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d2, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))






        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        # print "mask",target_mask.size()
        # print "t1",target.size()
        target = target[target_mask]
        # print "t2",target.size()
        if not target.data.dim():
            return Variable(torch.zeros(1))
        # print "p1", predict.size()
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        # print "p2", predict.size()


        # mask(10L, 321L, 321L)
        # t1(10L, 321L, 321L)
        # t2(1030410L, )
        # p1(10L, 21L, 321L, 321L)
        # p2(1030410L, 21L)
        # loss(1030410L, )
        # d(1030410L, )



        #predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #print "pred",predict.size()

        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average,reduce=False)




        # predict2=predict
        # target2=target
        #
        # n, c, h, w = predict2.size()
        # target_mask2 = (target2 >= 0) * (target2 != self.ignore_label)
        # target2= target2[target_mask2]
        # if not target2.data.dim():
        #     return Variable(torch.zeros(1))
        # predict2 = predict2.transpose(1, 2).transpose(2, 3).contiguous()
        # predict2 = predict2[target_mask2.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #
        #
        # D_out1=D_out.copy()
        # D_out1=Variable(torch.FloatTensor(D_out1)).cuda()
        # D_out3=((D_out1.view(n, h, w, 1).repeat(1, 1, 1, c))[target_mask2.view(n, h, w, 1).repeat(1, 1, 1, c)]).view(-1, c)
        # log_s=F.log_softmax(predict2)
        # logs2=log_s*D_out3
        # loss2=F.nll_loss(logs2, target2, weight, size_average=self.size_average, ignore_index=-100)
        return loss,target_mask


class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss
