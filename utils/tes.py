import torch
import numpy as np
from torch.autograd import Variable
ones=np.array([[1,2,3],[4,5,6]])
print ones.shape
ones=Variable(torch.Tensor(ones))
two=ones.view(2,3,1).repeat(1,1,5)

print ones
print two