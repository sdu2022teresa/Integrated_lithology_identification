import torch
import torch.nn as nn
import torch.nn.functional as F
# from .unet_base import *

from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out

class PAM_Module(nn.Module):
	"""Position attention module"""
	# Ref from SAGAN
	def __init__(self, in_dim):
		super(PAM_Module, self).__init__()
		self.channel_in = in_dim

		self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8,kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8,kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros(1))
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		"""
			inputs:
				x : input feature maps
			returns:
				out:attention value + input feature
				attention: B * (H*W) * (H*W)
		"""
		m_batchsize, C, height, width = x.size()#channel first
		proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
		proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
		energy = torch.bmm(proj_query,proj_key)
		attention = self.softmax(energy)
		proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

		out = torch.bmm(proj_value,attention.permute(0,2,1))
		out = out.view(m_batchsize, C, height, width)

		out = self.gamma*out + x

		return out


class CAM_Module(nn.Module):
	""" Channel attention module"""

	def __init__(self, in_dim):
		super(CAM_Module, self).__init__()
		self.chanel_in = in_dim

		self.gamma = nn.Parameter(torch.zeros(1))
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		"""
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
		m_batchsize, C, height, width = x.size()
		# A -> (N,C,HW)
		proj_query = x.view(m_batchsize, C, -1)
		# A -> (N,HW,C)
		proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
		# X -> (N,C,C)
		energy = torch.bmm(proj_query, proj_key)
		energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

		attention = self.softmax(energy_new)
		# A -> (N,C,HW)
		proj_value = x.view(m_batchsize, C, -1)
		# XA -> （N,C,HW）
		out = torch.bmm(attention, proj_value)
		# output -> (N,C,H,W)
		out = out.view(m_batchsize, C, height, width)

		out = self.gamma * out + x
		return out

