from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DehazeGAN(nn.Module):
	def __init__(self, channel):
		super(DehazeGAN, self).__init__()
        
		self.encoder_large = DehazeGan(channel)
		self.encoder_medium = DehazeGan(channel*2)
		self.encoder_small = DehazeGan(channel*4)
		
		self.decoder_small = DehazeGan(channel*4)
		self.decoder_medium = DehazeGan(channel*2)
		self.decoder_large = DehazeGan(channel)
        
		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)
        
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  
 
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')

	def forward(self,x):
		x_elin = self.ein(self.conv_in(x))
		elout = self.encoder_large(x_elin)

		x_emin = self.conv_eltem(self.maxpool(elout))      
		emout = self.encoder_medium(x_emin)

		x_esin = self.conv_emtes(self.maxpool(emout))          
		esout = self.encoder_small(x_esin)

		dsout = self.decoder_small(esout)

		x_dmin = self._upsample(self.conv_dstdm(dsout), emout) + emout
		dmout = self.decoder_medium(x_dmin)

		x_dlin = self._upsample(self.conv_dmtdl(dmout), elout) + elout    
		dlout = self.decoder_large(x_dlin)

		x_out = self.conv_out(dlout) + x

		return x_out


class DehazeGan:
	def __init__(self, conv_num_channels: int, emb_dim: int, num_heads: int) -> None:
		self.brb = BRB(conv_num_channels)
		self.color_attention = nn.MultiheadAttention(emb_dim, num_heads)

	def forward(self, x):
		y = self.brb(x)
		y = self.color_attention(y) + x
		return y

    
    
class BRB(nn.Module):
	def __init__(self,channel,norm=False):                                
		super(BRB,self).__init__()

		self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		#self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)

		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1(x)))
		x_2 = self.act(self.norm(self.conv_2(x_1)))
		x_out = self.act(self.norm(self.conv_out(x_2)) + x)

		return	x_out
    

        