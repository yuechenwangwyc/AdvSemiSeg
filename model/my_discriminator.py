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

		self.conv1 = nn.Conv2d(num_classes+3, ndf, kernel_size=4, stride=1, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=1, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=1, padding=1)
		self.conv4 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv5 = nn.Conv2d(ndf*4, ndf * 4, kernel_size=4, stride=2, padding=1)
		self.conv6 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
		self.conv7 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=1)
		self.classifier=nn.Linear(ndf*8*38*38,(num_classes-1)*2)
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
		x = self.conv6(x)
		x = self.leaky_relu(x)
		x = self.conv7(x)
		x = self.leaky_relu(x)

		x=x.view(x.size()[0],-1)
		x = self.classifier(x)

		x=self.sigmoid(x)



		return x

