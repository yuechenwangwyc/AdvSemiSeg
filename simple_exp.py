# import numpy as np
#
# #
#
# #m=np.array([[1,2,3],[4,5,6],[7,8,9]])
# # print m
# #
# # print np.amax(m,axis=1)
#
#
# # import torch
# #
# # x=torch.randn(3,4)
# # indices=torch.LongTensor([3])
# # print x
# # print x.repeat(1,2)
#
#
# s=set({1.0,0.01,0.0})
# print s
# s.discard(0.0)
# print s

import matplotlib.pyplot as plt
import numpy as np
import math
import torch

# x=torch.FloatTensor(np.arange(0,1,0.1))
# x=np.arange(0,1,0.1)
# x=torch.from_numpy(x)
#
# y=1/(math.e**(((x-0.53)*20)*(-1))+1)
# print y

x=np.arange(0,1,0.1)

y=1/(math.e**(((x-0.5)*30)*(-1))+1)


print y

t=x


plt.title("x")
plt.plot(x,y)
plt.plot(x,t)
plt.show()