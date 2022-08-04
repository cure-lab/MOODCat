import os
from os.path import join
import functools
import math
import numpy as np
use_tqdm=False
if use_tqdm:
	from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
####
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torch.utils.data as data
import torchvision.datasets as dset
import random
import gtsrb_dataset as dataset

# from tensorboardX import SummaryWriter

# Import my stuff
import inception_utils
import utils

from PyTorchDatasets import CocoAnimals, FFHQ, Celeba
from fid_score import calculate_fid_given_paths_or_tensor
from torchvision.datasets import ImageFolder
import pickle
from matplotlib import pyplot as plt
from mixup import CutMix
import gc
import sys
from types import ModuleType, FunctionType
from gc import get_referents

####


# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

#** yyj modified from biggan: datasets.py
import h5py as h5
import torch
class ILSVRC_HDF5(data.Dataset):
	def __init__(self, root, transform=None, target_transform=None,
							 load_in_mem=False, train=True,download=False, validate_seed=0,
							 val_split=0, **kwargs): # last four are dummies
			
		self.root = root
		self.num_imgs = len(h5.File(root, 'r')['labels_train'])
		
		# self.transform = transform
		self.target_transform = target_transform   
		
		# Set the transform here
		self.transform = transform
		
		# load the entire dataset into memory? 
		self.load_in_mem = load_in_mem
		
		# If loading into memory, do so now
		if self.load_in_mem:
			print('Loading %s into memory...' % root)
			with h5.File(root,'r') as f:
				self.data = f['imgs_train'][:].transpose([0,3,1,2])
				self.labels = f['labels_train'][:]

	def __getitem__(self, index):
		"""
		Args:
				index (int): Index

		Returns:
				tuple: (image, target) where target is class_index of the target class.
		"""
		# If loaded the entire dataset in RAM, get image from memory
		if self.load_in_mem:
			img = self.data[index]
			target = self.labels[index]
		
		# Else load it from disk
		else:
			with h5.File(self.root,'r') as f:
				img = f['imgs_train'][index]
				target = f['labels_train'][index]
		
	 
		# if self.transform is not None:
		#     img = self.transform(img)
		# Apply my own transform
		img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
		
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		return img, int(target)

	def __len__(self):
			return self.num_imgs
			# return len(self.f['imgs'])

class CenterCropLongEdge(object):
	"""Crops the given PIL Image on the long edge.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
	"""
	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.
		Returns:
			PIL Image: Cropped image.
		"""
		return transforms.functional.center_crop(img, min(img.size))

	def __repr__(self):
		return self.__class__.__name__

class RandomCropLongEdge(object):
	"""Crops the given PIL Image on the long edge with a random start point.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
	"""
	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be cropped.
		Returns:
			PIL Image: Cropped image.
		"""
		size = (min(img.size), min(img.size))
		# Only step forward along this edge if it's the long edge
		i = (0 if size[0] == img.size[0] 
				else np.random.randint(low=0,high=img.size[0] - size[0]))
		j = (0 if size[1] == img.size[1]
				else np.random.randint(low=0,high=img.size[1] - size[1]))
		return transforms.functional.crop(img, i, j, size[0], size[1])

	def __repr__(self):
		return self.__class__.__name__


def run(config):

	import train_fns

	config['resolution'] = 32#utils.imsize_dict[config['dataset']]
	print("RESOLUTION: ",config['resolution'])
	config['n_classes'] = 10#utils.nclass_dict[config['dataset']]
	config['G_activation'] = utils.activation_dict[config['G_nl']]
	config['D_activation'] = utils.activation_dict[config['D_nl']]
	# By default, skip init if resuming training.
	if config['resume']:
		print('Skipping initialization for training resumption...')
		config['skip_init'] = True
	config = utils.update_config_roots(config)
	device = 'cuda'
	# Seed RNG
	utils.seed_rng(config['seed'])
	# Prepare root folders if necessary
	utils.prepare_root(config)
	# Setup cudnn.benchmark for free speed, but only if not more than 4 gpus are used
	if "4" not in config["gpus"]:
		torch.backends.cudnn.benchmark = True
	print(":::::::::::/nCUDNN BENCHMARK", torch.backends.cudnn.benchmark, "::::::::::::::" )
	# Import the model--this line allows us to dynamically select different files.
	model = __import__(config['model'])
	state_dict = {'itr': 0, 'epoch': 13, 'save_num': 0, 'save_best_num': 0,
						'best_IS': 0,'best_FID': 999999,'config': config}
	G = model.Generator(**config).to(device)

	D = model.Unet_Discriminator(**config).to(device)
	encoder = model.Encoder(isize=32, nz=128, nc=3, ndf=32).to(device)
	# If using EMA, prepare it
	if config['ema']:
		print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
		G_ema = model.Generator(**{**config, 'skip_init':True,
									 'no_optim': True}).to(device)
		ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
	else:
		G_ema, ema = None, None


	GD = model.G_D(G, D, encoder, config)

	# print('Number of params in G: {} D: {} encoder:{}'.format(
	# *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))


	G_batch_size = 16
	if config['resume']:
		print('Loading weights...')
		if config["epoch_id"] !="":
			epoch_id = config["epoch_id"]
		experiment_name="test_synthesis"
		try:
			print("LOADING EMA")

			# TODO: utils'load_weights needs to add encoder part
			utils.load_weights(G, D, encoder, state_dict,
							config['weights_root'], experiment_name, config, epoch_id,
							config['load_weights'] if config['load_weights'] else None,
							G_ema if False else None)
		except:
			print("Ema weight wasn't found, copying G weights to G_ema instead")
			utils.load_weights(G, D, encoder, state_dict,
							config['weights_root'], experiment_name, config, epoch_id,
							config['load_weights'] if config['load_weights'] else None,
							 None)
			G_ema.load_state_dict(G.state_dict())

		print("loaded weigths")



	# If parallel, parallelize the GD module
	if config['parallel']:
		GD = nn.DataParallel(GD)
		encoder = nn.DataParallel(encoder)

	batch_size = config['batch_size']

	root = config["data_folder"]
	root_perm = config["data_folder"]
	# dataset = ILSVRC_HDF5(root=root, transform=None, load_in_mem=True)#hdf5 has already done the tranforms
	# transforms =  torchvision.transforms.Resize([256,256])
	norm_mean = [0.5,0.5,0.5]
	norm_std = [0.5,0.5,0.5]


	data_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
	data_transform = transforms.Compose([CenterCropLongEdge(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), ( 0.5, 0.5, 0.5))])
	# test_dataset = torchvision.datasets.SVHN(root='"/data/yangyijun/projects/adaptive attacks/dataset', split='test', transform = data_transform, download=True)

	# test_dataset = torchvision.datasets.CIFAR10(root=root, transform=data_transform, download=True, train=False)
	test_dataset = dataset.GTSRB(
            root_dir='/data/yangyijun/GTSRB_masking/', train=True, transform=data_transform
            )




	data_loader = DataLoader(test_dataset,batch_size*config["num_D_accumulations"],drop_last=True,num_workers=32, pin_memory=True, shuffle=True)
	loaders = [data_loader]

	for epoch in range(state_dict['epoch'], config['num_epochs']):
		if config["progress_bar"]: # * True
			if config['pbar'] == 'mine':
				pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
			else:
				pbar = tqdm(loaders[0])
		else:
			pbar = loaders[0]

		target_map = None



		for i, batch_data in enumerate(pbar):# * the i indicates one batch data
			x = batch_data[0].cuda()
			y = batch_data[1].cuda()
			state_dict['itr'] += 1

			G.eval()
			D.eval()
			encoder.eval()    
			# z_mean, z_log_var, z_ = encoder(x)
			checkpoint_name = "ep_627"
			zeros = torch.ones_like(y).cuda()
			ones = torch.ones_like(y).cuda()
			reference_x = x.view(x.shape[0], 3, 8, 4, 8, 4)
			del x
			reference_x = reference_x.permute(0, 2, 1, 4, 3, 5)
			reference_x = reference_x.permute(0, 1, 3, 2, 4, 5)
			reference_x = reference_x.permute(1, 2, 0, 3, 4, 5)
			h_size, w_size, B, C, H, W = reference_x.shape
			l = [i for i in range(w_size*h_size)]
			# random.shuffle(l)
			reference_x = reference_x.reshape(h_size*w_size, B, C, -1)
			# reference_x[l[0:128]] = 0 # mask ratio = 0.75
			reference_x = reference_x.permute(1, 2, 3, 0)
			reference_x = reference_x.reshape(B, C, H, W, h_size, w_size)
			reference_x = torch.cat([reference_x[..., i] for i in range(w_size)], 3)
			reference_x = torch.cat([reference_x[..., i] for i in range(h_size)], 2)
			erasing_transforms = transforms.RandomErasing(p=1, scale=(0.1, 0.33), ratio=(0.5,2.5))
			reference_x = erasing_transforms(reference_x)
			_, _, z = encoder(reference_x)
			torchvision.utils.save_image(reference_x[0:64], "./0124_masking_generator_results/"+checkpoint_name+"_GTSRB_real_"+str(i)+"_50per.png", nrow=8, normalize=True)
			del reference_x 
			# synthesis = G(z, G.shared(y))
			# torchvision.utils.save_image(synthesis[0:64], "./0124_masking_generator_results/"+checkpoint_name+"_GTSRB_synthesis_"+str(i)+"_50per.png", nrow=8, normalize=True)
			# del synthesis
			ones_synthesis = G(z, G.shared(ones*2))
			ones_synthesis = ones_synthesis.data.cpu()
			torchvision.utils.save_image(ones_synthesis[0:64], "./0124_masking_generator_results/"+checkpoint_name+"_GTSRB_ones_synthesis_"+str(i)+"_50per.png", nrow=8, normalize=True)
			del ones_synthesis 
			zeros_synthesis = G(z, G.shared(zeros*4))
			zeros_synthesis = zeros_synthesis.data.cpu()
			torchvision.utils.save_image(zeros_synthesis[0:64], "./0124_masking_generator_results/"+checkpoint_name+"_GTSRB_zeros_synthesis_"+str(i)+"_50per.png", nrow=8, normalize=True)
			del zeros_synthesis


			
			

		if i >= 10:
			break



		state_dict['epoch'] += 1

def main():

	# parse command line and run
	parser = utils.prepare_parser()
	config = vars(parser.parse_args())

	if config["gpus"] !="":
		os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]

	keys = sorted(config.keys())




	run(config)
if __name__ == '__main__':
	main()


