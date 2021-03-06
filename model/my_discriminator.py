import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Conv2d(num_classes+3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x

class Discriminator2(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator2, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x

class Discriminator2_con_pred(nn.Module):

	def __init__(self, num_classes, ndf = 90):
		super(Discriminator2_con_pred, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*4, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x


class Discriminator2_mul2(nn.Module):

	def __init__(self, num_classes, ndf = 32):
		super(Discriminator2_mul2, self).__init__()

		self.conv1 = nn.Conv2d(3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x


class Discriminator2_mul(nn.Module):

	def __init__(self, num_classes, ndf = 32):
		super(Discriminator2_mul, self).__init__()

		self.conv1 = nn.Conv2d(3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*12*12,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		#print x.size()
		x=x.view(x.size()[0],-1)
		#
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x

class Discriminator2_patch(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator2_patch, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		#self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)#321
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)

		x = self.conv6(x)
		x = self.leaky_relu(x)
		#print x.size()
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		#x=x.view(x.size()[0],-1)
		#x = self.classifier(x)
		x=self.sigmoid(x)


		return x

class Discriminator2_patchs(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator2_patchs, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=1, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		#self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)#321
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)

		x = self.conv6(x)
		x = self.leaky_relu(x)
		#print x.size()
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		#x=x.view(x.size()[0],-1)
		#x = self.classifier(x)
		x=self.sigmoid(x)


		return x

class Discriminator2_patch3(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator2_patch3, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(ndf*16, 1, kernel_size=4, stride=1, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		#self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)#321
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)

		x = self.conv6(x)
		x = self.leaky_relu(x)
		#print x.size()
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		#x=x.view(x.size()[0],-1)
		#x = self.classifier(x)
		x=self.sigmoid(x)


		return x


class Discriminator2_patch2(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator2_patch2, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		#self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)#321
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		#print x.size()
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		#x=x.view(x.size()[0],-1)
		#x = self.classifier(x)
		x=self.sigmoid(x)


		return x



class Discriminator2_dis2(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator2_dis2, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		x = self.conv6(x)
		x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x


class Discriminator3(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator3, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*4, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x

class Discriminator_concat(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator_concat, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*2+3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x


class Discriminator_n(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator_n, self).__init__()

		self.conv1 = nn.Conv2d(4, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*39*39,1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.sigmoid=nn.Sigmoid()



	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)
		x=x.view(x.size()[0],-1)
		x = self.classifier(x)
		x=self.sigmoid(x)


		return x





class Discriminator_mul(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator_mul, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier = nn.Linear(ndf * 8 * 39 * 39, 40)
		self.sigmoid = nn.Sigmoid()

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)




	def forward(self, x):

		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)

		x=x.view(x.size()[0],-1)

		x = self.classifier(x)


		return x


class Discriminator_mul2(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(Discriminator_mul2, self).__init__()

		self.conv1 = nn.Conv2d(num_classes*3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		# self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		# self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier = nn.Linear(ndf * 8 * 39 * 39, 40)
		self.sigmoid = nn.Sigmoid()

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)




	def forward(self, x):

		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.conv5(x)
		x = self.leaky_relu(x)
		# x = self.conv6(x)
		# x = self.leaky_relu(x)
		# x = self.conv7(x)
		# x = self.leaky_relu(x)

		x=x.view(x.size()[0],-1)

		x = self.classifier(x)


		return x

