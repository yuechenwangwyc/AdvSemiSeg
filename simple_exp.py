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

# x=np.array([[0,1,6],[1,4,5]])
# print x
# y=(x==1)
# print y
# x[y]=2
# print x




# x=torch.FloatTensor(np.arange(0,1,0.1))
# x=np.arange(0,1,0.1)
# x=torch.from_numpy(x)
#
# y=1/(math.e**(((x-0.53)*20)*(-1))+1)
# print y




'''
x=np.arange(0,1.5,0.1)

# y=1/(math.e**(((x-0.6)*30)*(-1))+1)

y=np.sin((x-0.3)*1.3)*1.3
z=np.sin((0.005-0.3)*1.3)*1.3
print z

print y

k=np.cos((1-0.3)*1.3)*1.3*1.3
print k

t=x


plt.title("x")
plt.plot(x,y)
plt.plot(x,t)
plt.show()
'''

import numpy as np

mat=np.ones([600,600])*(0)

import cv2

cv2.imwrite('/data1/wyc/0.png', mat)
