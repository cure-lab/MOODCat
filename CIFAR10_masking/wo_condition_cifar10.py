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
import gtsrb_dataset as dataset
from tensorboardX import SummaryWriter

# Import my stuff
import inception_utils
import wo_condition_utils as utils

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
        self.data = f['imgs_train'][:] #.transpose([0,3,1,2])
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
        img = f['imgs_train'][index]#.transpose([2,0,1])
        target = f['labels_train'][index]
    # Apply my own transform
    # img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2    
   
    if self.transform is not None:
        img = self.transform(img)

    
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

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

torch.backends.cudnn.benchmark = True

def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def run(config):

    import wo_condition_train_fns as train_fns

    config['resolution'] = 32#utils.imsize_dict[config['dataset']]
    print("RESOLUTION: ",config['resolution'])
    # exit()
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
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    
    # * here we define the tensorboard indicator
    # TODO: Need the local_rank ?
    writer = SummaryWriter(log_dir=join('./logs', experiment_name))

    print('Experiment name is %s' % experiment_name)
    print("::: weights saved at ", '/'.join([config['weights_root'],experiment_name]) )
    # Next, build the model
    keys = sorted(config.keys())
    for k in keys:
        print(k, ": ", config[k])
    G = model.Generator(**config).to(device)
    encoder = model.Encoder(isize=32, nz=128, nc=3, ndf=32).to(device)
    D = model.Unet_Discriminator(**config).to(device)

    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init':True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None

    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()

    GD = model.G_D(G, D, encoder, config)

    print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))

    # Prepare noise and randomly sampled label arrays Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    G_batch_size = int(G_batch_size*config["num_G_accumulations"])
    # z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'], # ** this z_ and y_ pair is random sampled. 
    #                        device=device, fp16=config['G_fp16'])
    # ** z_ and y_ cannot be prepared before data coming
    # TODO: randomly samples some z to training the cGAN, akin to the trunction trick


    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0,'best_FID': 999999,'config': config}
    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        if config["epoch_id"] !="":
            epoch_id = config["epoch_id"]

        try:
            print("LOADING EMA")
            utils.load_weights(G, D, encoder, state_dict,
                            config['weights_root'], experiment_name, config, epoch_id,
                            config['load_weights'] if config['load_weights'] else None,
                            G_ema if config['ema'] else None)
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
        # G = nn.DataParallel(G)
        # D = nn.DataParallel(D)
    if config['cross_replica']: #* default= False
        # TODO: need to check
        patch_replication_callback(GD)
        # patch_replication_callback(encoder)
    # Prepare loggers for stats; metrics holds test metrics, lmetrics holds any desired training metrics.
    
    # TODO: Notice this part test_metrics_fname
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                 reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed to the dataloader, as G doesn't require dataloading. Note
    # that at every loader iteration we pass in enough data to complete a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])

# TODO: modify the data augmentation part
    if config["dataset"]=="CIFAR10":# imageNet100
        batch_size = config['batch_size']
        root = config["data_folder"]
        
        data_transform = transforms.Compose([CenterCropLongEdge(), transforms.Resize(32), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), ( 0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root=root, transform=data_transform, download=True, train=True)
        test_dataset = torchvision.datasets.CIFAR10(root=root, transform=data_transform, download=True, train=False)

        train_dataset_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size*config["num_D_accumulations"],drop_last=True,num_workers=32, pin_memory=True, shuffle=True)
            
        test_dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size*config["num_D_accumulations"],drop_last=True,num_workers=32, pin_memory=True, shuffle=True)
        loaders = [train_dataset_loader]

    print("Loaded ", config["dataset"])
    # inception_metrics_dict = {"fid":[],"is_mean": [], "is_std": []}


    # Prepare inception metrics: FID and IS
    # get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'],config['parallel'], config['no_fid'], use_torch=False)

    # Prepare a fixed z & y to see individual sample evolution throghout training
    # fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
    #                                    config['n_classes'], device=device,
    #                                    fp16=config['G_fp16'])
    # fixed_z.sample_()
    # fixed_y.sample_()
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                            device=device, fp16=config['G_fp16'])
    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN': # which_train_fn ==GAN
        train = train_fns.GAN_training_function(G, D, GD, encoder,
                                                ema, state_dict, config, writer=writer)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    # ** this sample might be useless -> return G_z and y_
    # sample = functools.partial(utils.sample,
    #                       G=(G_ema if config['ema'] and config['use_ema']
    #                          else G),
    #                       z_=z_, y_=y_, config=config)



    if config["debug"]:
        loss_steps = 10
    else:
        loss_steps = 100

    print('Beginning training at epoch %d...' % state_dict['epoch'])


    # Train for specified number of epochs, although we mostly track G iterations.
    warmup_epochs = config["warmup_epochs"] #* warmup_epochs=200


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
            x = batch_data[0]
            y = batch_data[1]
            #H = batch_data[2]
            # print(x.shape)
            # exit()
		    


            # Increment the iteration counter
            state_dict['itr'] += 1
            if config["debug"] and state_dict['itr']>config["stop_it"]:
                print("code didn't break :)")
                #exit(0)
                break #better for profiling
            # Make sure G and D are in training mode, just in case they got set to eval For D, which typically doesn't have BN, this shouldn't
            # matter much.
            G.train()
            D.train()
            encoder.train()
            if config['ema']: # * ema==True
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device).view(-1)
            else:
                x, y = x.to(device), y.to(device).view(-1)
            x.requires_grad = False
            y.requires_grad = False



            if config["unet_mixup"]:
                # Here we load cutmix masks for every image in the batch
                n_mixed = int(x.size(0)/config["num_D_accumulations"])
                target_map = torch.cat([CutMix(config["resolution"]).cuda().view(1,1,config["resolution"],config["resolution"]) for _ in range(n_mixed) ],dim=0)


            if config["slow_mixup"] and config["full_batch_mixup"]: # * full_batch_mixup==True slow_mixup==True
                # r_mixup is the chance that we select a mixed batch instead of
                # a normal batch. This only happens in the setting full_batch_mixup.
                # Otherwise the mixed loss is calculated on top of the normal batch.
                r_mixup = 0.5 * min(1.0, state_dict["epoch"]/warmup_epochs) # r is at most 50%, after reaching warmup_epochs
            elif not config["slow_mixup"] and config["full_batch_mixup"]:
                r_mixup = 0.5
            else:
                r_mixup = 0.0

            metrics = train(x, y, state_dict["epoch"], batch_size , target_map = target_map, r_mixup = r_mixup, writer=writer, step_count=i) # * mixup_rate=0.5

            # print(i) #* here the i indicates the iteration
            if (i+1)%200==0:
                # print this just to have some peace of mind that the model is training
                print("alive and well at ", state_dict['itr'])

            if (i+1)%20==0:
                #try:
                # * here thei "itr" indicates the epoch number.
                train_log.log(itr=int(state_dict['itr']), **metrics)
                #except:
                #    print("ouch")
            # Every sv_log_interval, log singular values # * sv_log_interval=10
            # TODO: check this log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):

                train_log.log(itr=int(state_dict['itr']),
                             **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

          # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):

                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    encoder.eval()
                    if config['ema']:
                        G_ema.eval()
                    train_fns.save_and_sample(G, D,  encoder, G_ema, x, y, 
                                      state_dict, config, experiment_name, sample_only=False)

            go_ahead_and_sample = (not (state_dict['itr'] % config['sample_every']) ) or ( state_dict['itr']<1001 and not (state_dict['itr'] % 100) )

            if go_ahead_and_sample:

                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    encoder.eval()
                    if config['ema']:
                        G_ema.eval()

                    train_fns.save_and_sample(G, D, encoder, G_ema, x, y,
                                      state_dict, config, experiment_name, sample_only=True)
                    print("saved models")
                    # TODO: Enable this part later
                    # with torch.no_grad():
                    #     real_batch = dataset.fixed_batch()
                    # train_fns.save_and_sample(G, D, encoder, G_ema, z_, y_, fixed_z, fixed_y,
                    #                   state_dict, config, experiment_name, sample_only=True, use_real = True, real_batch = real_batch)

                    # also, visualize mixed images and the decoder predicitions
                    
                    # if config["unet_mixup"]:
                    #     with torch.no_grad():

                    #         n = int(min(target_map.size(0), x.size(0)/2))
                    #         which_G = G_ema if config['ema'] and config['use_ema'] else G
                    #         utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                    #                                                      z_, y_, config['n_classes'],
                    #                                                      config['num_standing_accumulations'])

                    #         if config["dataset"]=="coco_animals":
                    #             real_batch, real_y = dataset.fixed_batch(return_labels = True)

                    #             fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z[:n], which_G.shared(real_y[:n])))
                    #             mixed = target_map[:n]*real_batch[:n]+(1-target_map[:n])*fixed_Gz
                    #             train_fns.save_and_sample(G, D, G_ema, z_[:n], y_[:n], fixed_z[:n], fixed_y[:n],
                    #                         state_dict, config, experiment_name+"_mix", sample_only=True, use_real = True, real_batch = mixed, mixed=True, target_map = target_map[:n])

                    #         else:
                    #             # real_batch = dataset.fixed_batch()
                    #             # fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z[:n], which_G.shared(fixed_z[:n]))) #####shouldnt that be fixed_y?

                    #             # mixed = target_map[:n]*real_batch[:n]+(1-target_map[:n])*fixed_Gz
                    #             # train_fns.save_and_sample(G, D, G_ema, z_[:n], y_[:n], fixed_z[:n], fixed_y[:n],
                    #             #             state_dict, config, experiment_name+"_mix", sample_only=True, use_real = True, real_batch = mixed, mixed=True, target_map = target_map[:n])
                    #             pass
                    #             # TODO: enable the visualization

          # Test every specified interval
            '''
            if not (state_dict['itr'] % config['test_every']):
            #if state_dict['itr'] % 100 == 0:
                if config['G_eval_mode']:
                  print('Switchin G to eval mode...')

                is_mean, is_std , fid = train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics , experiment_name, test_log, moments = "train")
                ###
                #  Here, the bn statistics are updated
                ###
                if  config['accumulate_stats']:
                    print("accumulate stats")
                    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                                                 z_, y_, config['n_classes'], config['num_standing_accumulations'])

                inception_metrics_dict["is_mean"].append((state_dict['itr'] , is_mean ) )
                inception_metrics_dict["is_std"].append((state_dict['itr'] , is_std ) )
                inception_metrics_dict["fid"].append((state_dict['itr'] , fid ) )

            if (i + 1) % loss_steps == 0:
                with open(os.path.join(config["base_root"],"logs/inception_metrics_"+config["random_number_string"]+".p"), "wb") as h:
                    pickle.dump(inception_metrics_dict,h)
                    print("saved FID and IS at", os.path.join(config["base_root"],"logs/inception_metrics_"+config["random_number_string"]+".p") )
            '''
            G.train()
            # encoder.module.train()
            D.train()
        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1

def main():

    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    if config["gpus"] !="":
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    random_number_string = str(int(np.random.rand()*1000000)) + "_" + config["id"]
    config["stop_it"] = 99999999999999

    config["debug"]= False
    if config["debug"]:
        config["save_every"] = 30
        config["sample_every"] = 20
        config["test_every"] = 20
        config["num_epochs"] = 1
        config["stop_it"] = 35
        config["slow_mixup"] = False

    config["num_gpus"] = len(config["gpus"].replace(",",""))

    config["random_number_string"] = random_number_string
    new_root = os.path.join(config["base_root"],random_number_string)
    if not os.path.isdir(new_root):
        os.makedirs(new_root)
        os.makedirs(os.path.join(new_root, "samples"))
        os.makedirs(os.path.join(new_root, "weights"))
        os.makedirs(os.path.join(new_root, "data"))
        os.makedirs(os.path.join(new_root, "logs"))
        print("created ", new_root)
    config["base_root"] = new_root


    keys = sorted(config.keys())
    print("config")
    for k in keys:
        print(str(k).ljust(30,"."), config[k] )



    run(config)
if __name__ == '__main__':
    main()
