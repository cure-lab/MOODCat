import numpy as np
import math
import functools
import torch
import torch.nn import init  
import torch.optim as optim
import torch.nn.functional as F 
import torch.nn import Parameter as P 

import layers 

import utils 
import copy
from matplotlib import pyplot as plt

def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
	arch = {}
	arch[128] = {'in_channels': [ch*item for item in [16, 16, 8, 4, 2]],
					'out_channels':[ch*item for item in[16, 8, 4, 2, 1]],
					'upsample': [True]*5,
					'resolution': [8, 16, 32, 64, 128],
					'attation': {2**i: (2**i in [int(item) for item in attention.split('_')]) for i in range(3,8)}}
	return arch 

class Generator(nn.Module):
	def __init__(self, G_CH=64, dim_z=128, bottom_width=4, resolution=128, 					G_kernel_size=3, G_attn='64', n_classes=100,
				num_G_SVs=1, num_G_SV_itrs=1,
				G_shared=True, shared_dim=0, hier=False,
				cross_replica=False, mybn=False,
				G_activation=nn.ReLU(inplace=False),
				G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8, 
				BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False,
				G_init='ortho', skip_init=False, no_optim=False,
				G_param='SN', norm_style='bn',
				**kwargs):
		super(Generator, self).__init__()
		# channel width mulitplier
		self.ch = G_ch
		# Dimensionality of the latent space
		self.dim_z = dim_z
		# The initial spatial dimensions
		self.bottom_width = bottom_width
		# Resolution of the output
		self.resolution = resolution
		# Kernel_size?
		self.kernel_size = G_kernel_size
		# Attention?
		self.attention = G_attn
		# number of class, for use in categorical conditional generation
		self.n_classes = n_classes
		# Use shared embeddings?
		self.G_shared = G_shared
		# Dimensionality of the shared embedding? Unused if not using G_shared
		self.shared_dim = shared_dim if shared_dim > 0 else dim_z
		# Hierarchical latent space?
		self.hier = hier
		# Cross replica batchnorm?
		self.cross_replica = cross_replica
		# Use my batchnorm?
		self.mybn = mybn
		# nonlinerity for residual blocks
		self.activation = G_activation
		# Initialization style
		self.init = G_init 
		# Parameterization style
		self.G_param = G_param 
		# Normalization style
		self.norm_style = norm_style
		# Epsilon for BatchNorm?
		self.BN_eps = BN_eps
		# Eplison for Spectral Norm?
		self.SN_eps = SN_eps
		# fp16?
		self.fp16 = G_fp16
		# Architecture dict 
		self.arch = G_arch(self.ch, self.attention)[resolution]
		self.unconditional = kwargs["unconditional"]# **kwargs is a keyword dict


		# If using hierarchical latents, adjust z #! the hierachical here indicates the chunked z into deep layer
		if self.hier:
			# Number of places z slots into 
			self.num_slots = len(self.arch['in_channels']) + 1
			self.z_chunk_size = (self.dim_z // self.num_slots)

			if not self.unconditional:# if conditional
				self.dim_z = self.z_chunk_size * self.num_slots
		else:
			self.num_slots = 1
			self.z_chunk_size = 0

		# which convs, batchnorms, and linera layers to use
		if self.G_param == 'SN':
			self.which_conv = functools.partial(layers.SNConv2d, 		kernal_size=3, padding=1, num_svs=num_G_SVs, num_itrs=num_G_SVitrs, eps=self.SN_eps)# rewrite the keywords of SNConv2d
			self.which_linear = functools.partial(layer.SNLinear, num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps)
		else:
			self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
			self.which_linear = nn.Linear

		# We use a non-spectral-normed embedding here regardless;
		# For some reason applying SN to G's embedding seems to randomly cripple G
		self.which_embedding = nn.Embedding

		if self.unconditional: # conditional = False
			bn_linear = nn.Linear 
			input_size = self.dim_z + (self.shared_dim if self.G_shared else 0 )
		else: # condition==True
			bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared else self.which_embedding)

			input_size = (self.shared_dim + self.z_chunk_size if self.G_shared else self.n_classes)

		self.which_bn = functools.partial(layers.ccbn,
		#TODO: reading the layers.ccbn 
		which_linear=bn_linear, cross_replica=self.cross_replica, mybn=self.mybn, input_size=input_size, norm_style=self.norm_style, eps=self.BN_eps, self_modulation=self.uncoditioanal)


		# Prepare model

		# If not using shared embeddings, self.shared is just a passthrough 
		self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 													else layers.identity())
		# First linear layer
		if self.unconditional: # conditional==False
			self.linear = self.which_linear(self.dim_z, self.arch['in_channels'][0] * (self.bottom_width**2))
		else:
			self.linear = self.which_linear(self.dim_z//self.num_slots, self.arch['inchannels'][0]*(self.bottom_width**2))
		
		# self.blocks is a doubly-nested list of modules, the outer loop intended to be over blocks ata a given resolution (resblocks and/or self-attention), while the inner loop is over a given block

		self.blocks = []
		for index in range(len(self.arch['out_channels'])):
			
			self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index], out_channels=self.arch['out_channels'][index], which_conv=self.which_conv,
			which_bn=self.which_bn,
			activation=self.activation,
			upsample=(functools.partial(F.interpolate, scale_factor=2) if self.arch['upsample'][index] else None))]]
			

			# if attention on this block, attach it to the end
			if self.arch['attention'][self.arch['resolution'][index]]:
				print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
				self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]



		#Turn self.blocks into a ModuleList so that it's all properly registered.
		self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

		# output layer: batchnorm-relu-conv.
		# Consider using a non-spectral conv here
		self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
			cross_replica=self.cross_replica, mybn=self.mybn), self.activation, self.which_conv(self.arch['out_channels'][-1], 3))

		#Initialize weights. Optionally skip init for testing.
		if not skip_init:
			self.init_weights()


		# Set up optimizer
		# If this is an EMA copy, no need for an optim, so just return now
		if no_optim:
			return 

		self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
		if G_mixed_precision:
			print('Using fp16 adam in G...')
			import utils
			self.optim = utils.Adam16(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

		else:
			self.optim = utils.Adam(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2),weight_decay=0, eps=self.adam_eps)


		# Initialize
		def inin_weights(self):
			self.param_count = 0
			for module in self.modules():
				if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
					if self.init == 'ortho':
						init.orthogonal_(module.weight)
					elif self.init == 'N02':
						init.normal_(module.weight, 0, 0.02)
					elif self.init in ['glort','xavier']:
						init.xavier_uniform_(module.weigh)
					else:
						print('Init style not recognized...')

					self.param_count += sum([p.data.nelement() for p in module.parameters()])

			print('Param count for G''s initialized parameters: %d' % self.param_count)


		# Note on this forward function: we pass in a y vector which has already been passed through G.shared to enable easy class-wise interpolation later. If we passed int the one-hot and then ran it through G.shared in this forward function, it would be harder to handle.

		def forward(self, z, y):
			# If hierachical, concatenate zs and ys
			if self.hier:
				# faces
				if self.uncoditional:
					ys = [z for _ in range(self.num_slots)]
				else: 
					zs = torch.split(z, self.z_chunk_size, dim=1)
					z = zs[0]
					ys = [torch.cat([y, item], dim=1) for item in zs[1:]]

			else:
				if self.unconditional:
					ys = [None] * len(self.blocks)
				else:
					ys = [y]*len(self.blocks)





			# First linear layer
			h = self.linear(z)
			# Reshape
			h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)

			# Loop over blocks
			for index, blocklist in enumerate(self.blocks):
				# Second inner loop in case block has multiple layers
				for block in blocklist:
					h = block(h, ys[index])
			# Apply batchnorm-relu-cov-tanh at output
			return torch.tanh(self.output_layer(h))

# Duscrunubatir arcgutectyrem same paradugn as G's above
def D_arch(ch=64, attention='64', ksize='33333', dilation='111111'):
	arch = {}
	arch[128] = {'in_channels': [3] + [ch*item for item in [1, 2, 4, 8, 16]],
					'out_channels': [item*ch for item in [1, 2, 4, 8, 16, 16]],
					'downsample': [True]*5 +[False],
					'resolution': [64, 32, 16, 8, 4, 4],
					'attention': {2**i: 2**i in [int(item) for item in attention.split('_')] for i in range(2,8)}}
	return arch 


def D_unet_arch(ch=64, attention='64', ksize='333333', dilation='111111', out_channel_multiplier=1):
	arch = {}
	n = 2
	ocm = out_channel_multiplier

	# covers bigger perceptual fields
	arch[128] = {'in_channels': [3] + [ch*item for item in [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2, 1]], 
				'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 8, 4, 2, 1, 1]], 'downsample':[True]*5 +[False]*5,
				'upsample':[False]*5 +[True]*5,
				'resolution': [64, 32, 16, 8, 4, 8, 16, 32, 64, 128], 
				'attention':{2**i: 2**i in [int(item) for item in attention.split('_')] for i in range(2, 11)}}
	return arch 

class Unet_Discriminator(nn.Module):

	def __init__(self, D_ch=64, D_wide=True, resoluton=128, 
							D_kernel_size=3, D_attn='64', n_classes=100,
							num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False), D_lr=2e-4, D_B1=0.0, D_B2=0.999,adam_eps=1e-8, DN_eps=1e-12, output_dim=1, D_mixed_precidion=False,
							D_fp16=False,
							D_init='orthon', skip_init=False, D_param='SN',
							decoder_skip_connection=True, **kwargs):
		super(Unet_Discriminator, self).__init__()

		# Width multiplier
		self.ch = D_ch
		# Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN 
		self.D_wide = D_wide 
		# Resolution
		self.resolution = resolution 
		# Kernel size 
		self.kernel_size = D_kernel_size
		# Attention?
		self.attention = D_attn 
		# Number of classes
		self.n_classes = n_classes
		# Activation
		self.activation = D_activation
		# initialization style
		self.init = D_init 
		# Parameterization style
		self.D_param = D_param 
		# Epsilon for Spectral Norm?
		self.SN_eps = SN_eps 
		# Fp16?
		self.fp16 = D_fp16 



		if self.resoluton==128:
			self.save_features =[0, 1, 2, 3, 4]
		self.out_channel_multiplier = 1# could be modified
		# Architecture
		self.arch = D_unet_arch(self.ch, self.attention, out_channel_multiplier = self.out_channel_multiplier)[resolution]

		self.unconditional = kwargs["unconditional"]
		# Which convs, batchnorms, and linear layers to use
		# No option to turn off SM in D right nwo

		if self.D_param == ''






# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
	def __init__(self, G, D, config):
		super(G_D, self).__init__()
		self.G = G
		self.D = D 
		self.config = config 
	def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z= False,
				split_D=False, dw1=[], dw2=[], reference_x=None, mixup=False, mixup_only=False, target_map=None):
		
		if mixup:
			gy=dy
			# why? so the mixup samples consisit of same class 
		
		# if training G, enable grad tape
		with torch.set_grad_enabled(train_G):
			G_z = self.G(z, self.G.shared(gy))
			# Cast as necessary
			if self.G.fp16 and not self.D.fp16:
				G_z = G_z.float()
			if self.D.fp16 and not self.G.fp16:
				G_z = G_z.half()

		if mixup:
			initial_x_size = x.size(0)
			mixed = target_map*x +(1-target_map)*G_z
			mixed_y = dy 

		




