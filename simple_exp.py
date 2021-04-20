import numpy as np

#

#m=np.array([[1,2,3],[4,5,6],[7,8,9]])
# print m
#
# print np.amax(m,axis=1)


import torch

x=torch.randn(3,4)
indices=torch.LongTensor([3])
print x
print x.repeat(1,2)