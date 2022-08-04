#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license) - some modifications
''' train_fns.py
Functions for the main loop of training different conditional image models
'''
from matplotlib import pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import utils
import losses
from PIL import Image
import numpy as np
import functools
import copy
import pytorch_ssim as ssim_package
# from tensorboardX import SummaryWriter

# Dummy training function for debugging
def dummy_training_function():
    def train(x, y):
        return {}
    return train

def BCEloss(D_fake, D_real, d_real_target, d_fake_target):
    real =  F.binary_cross_entropy_with_logits(D_real,d_real_target.expand_as(D_real))
    fake =  F.binary_cross_entropy_with_logits(D_fake,d_fake_target.expand_as(D_fake))
    return real, fake

def BCEfakeloss(D_fake,target):
    return F.binary_cross_entropy_with_logits(D_fake, target.expand_as(D_fake))

def GAN_training_function(G, D, GD, encoder, ema, state_dict, config, writer):
    def train(x, y, epoch, batch_size, target_map = None, r_mixup = 0.0, writer=writer, step_count=None):
        G.optim.zero_grad()
        D.optim.zero_grad()
        # encoder.optim.zero_grad()

        if config["unet_mixup"]:
            real_target = torch.tensor([1.0]).cuda()
            fake_target = torch.tensor([0.0]).cuda()

        if config["unet_mixup"] and not config["full_batch_mixup"]:
            use_mixup_in_this_round = True
        elif config["unet_mixup"] and config["full_batch_mixup"]: # *True
            use_mixup_in_this_round = torch.rand(1).detach().item()<r_mixup # * mixup's rate=0.5
        else:
            use_mixup_in_this_round = False

        out = {}

        skip_normal_real_fake_loss = (use_mixup_in_this_round and config["full_batch_mixup"] ) # * full_batch_mixup==True

        n_d_accu = config['num_D_accumulations'] # * num_D_accumulation=1 

        split_size = int(x.size(0)/n_d_accu)

        x = torch.split(x, split_size)
        y = torch.split(y, split_size)

        # import ipdb; ipdb.set_trace()
        d_real_target = torch.tensor([1.0]).cuda()
        d_fake_target = torch.tensor([0.0]).cuda()

        discriminator_loss = functools.partial(BCEloss, d_real_target=d_real_target, d_fake_target=d_fake_target)
        mix_fake_target = torch.tensor([1.0]).cuda()
        fake_loss = functools.partial(BCEfakeloss, target = mix_fake_target)

        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):# num D =1
            counter = 0
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()

            for accumulation_index in range(n_d_accu):#1
                # z_mean, z_var, z_ = encoder(x[counter])
                if use_mixup_in_this_round:
                    
                    # full_batch_mixup==True consistency_loss_and_augmentation=True consistency_loss=False
                    if (not config["full_batch_mixup"]) or (config["full_batch_mixup"] and (config["consistency_loss_and_augmentation"] or config["consistency_loss"]) ): # * go here
                        # split_D = False
                        D_fake, D_real , D_mixed, G_z, mixed,  D_middle_fake, D_middle_real, D_middle_mixed, target_map = GD(gy=y[counter],
                                                            x=x[counter], dy=y[counter], train_G=False,
                                                            split_D=config['split_D'], mixup = True, target_map = target_map) # mixup can be true because weight is set to 0 when no mixup is used
                    else:
                        D_mixed, G_z, mixed, D_middle_mixed, target_map = GD(gy=y_[:batch_size],
                                                            x=x[counter], dy=y[counter], train_G=False, return_G_z = True,
                                                            split_D=config['split_D'], mixup = True, mixup_only = True, target_map = target_map)

                    if config["slow_mixup"] and not config["full_batch_mixup"]:
                        mixup_coeff = min(1.0, epoch/config["warmup_epochs"])#use without full batch mixup
                    else:# * go here
                        mixup_coeff = 1.0
                    # ** set display_mixed_batch  ==  True
                    


                else:
                    D_fake, D_real , G_z, D_middle_fake, D_middle_real   = GD(gy=y[counter],
                                                            x=x[counter], dy=y[counter], train_G=False,
                                                        split_D=config['split_D'])



                if not skip_normal_real_fake_loss:
                    # * normal D trainng
                    D_loss_real_2d, D_loss_fake_2d = discriminator_loss(D_fake.view(-1), D_real.view(-1))
                    D_loss_real_2d_item = D_loss_real_2d.detach().item()
                    D_loss_fake_2d_item = D_loss_fake_2d.detach().item()

                if use_mixup_in_this_round  and (config["consistency_loss"] or config["consistency_loss_and_augmentation"]):
                    mix =  D_real*target_map + D_fake*(1-target_map)

                if use_mixup_in_this_round:

                    D_mixed_flattened = D_mixed.view(-1)
                    target_map_flattend = target_map.view(-1)

                    mix_list = []
                    for i in range(D_mixed.size(0)):
                        # MIXUP LOSS 2D
                        mix2d_i= F.binary_cross_entropy_with_logits(D_mixed[i].view(-1),target_map[i].view(-1) )
                        mix_list.append(mix2d_i)

                    D_loss_mixed_2d = torch.stack(mix_list)
                    #-> D_loss_mixed_2d.mean() is taken later

                    D_loss_mixed_2d_item = D_loss_mixed_2d.mean().detach().item()
                    #D_loss_mixed_2d = D_loss_mixed_2d.view(D_mixed.size()).mean([2,3])

                if not skip_normal_real_fake_loss: # * normal Dis training False
                    D_loss_real_middle, D_loss_fake_middle = discriminator_loss(D_middle_fake, D_middle_real)

                    D_loss_real_middle_item = D_loss_real_middle.detach().item()
                    D_loss_fake_middle_item = D_loss_fake_middle.detach().item()

                if use_mixup_in_this_round and not config["consistency_loss"]:
                    # consistency loss is only concerned with segmenter output

                    #target for mixed encoder loss is fake
                    mix_bce = F.binary_cross_entropy_with_logits(D_middle_mixed, fake_target.expand_as(D_middle_mixed), reduction="none")

                    mixed_middle_loss = mixup_coeff*mix_bce
                    mixed_middle_loss_item = mixed_middle_loss.mean().detach().item()

                if skip_normal_real_fake_loss:
                    D_loss_real = torch.tensor([0.0]).cuda()
                    D_loss_fake = torch.tensor([0.0]).cuda()
                else:
                    D_loss_real = D_loss_real_2d + D_loss_real_middle
                    D_loss_fake = D_loss_fake_2d + D_loss_fake_middle

                D_loss_real_item = D_loss_real.detach().item()
                D_loss_fake_item = D_loss_fake.detach().item()

                D_loss = 0.5*D_loss_real + 0.5*D_loss_fake

                if use_mixup_in_this_round:
                    if config["consistency_loss"] or config["consistency_loss_and_augmentation"]:
                        consistency_loss = mixup_coeff*1.0*F.mse_loss(D_mixed, mix )
                        consistency_loss_item = consistency_loss.float().detach().item()

                    if not config["consistency_loss"]:
                        # GAN loss from cutmix augmentation (=/= consitency loss)
                        mix_loss = D_loss_mixed_2d + mixed_middle_loss
                        mix_loss = mix_loss.mean()
                    else:
                        mix_loss = 0.0

                    if config["consistency_loss"]:
                        mix_loss = consistency_loss
                    elif config["consistency_loss_and_augmentation"]:
                        mix_loss = mix_loss + consistency_loss

                    D_loss = D_loss + mix_loss

                D_loss = D_loss / float(config['num_D_accumulations'])
                if step_count%200 == 0:
                    print("D_loss: %2f"%D_loss.detach().item())
                D_loss.backward()
                counter += 1

            # Optionally apply ortho reg in D
            if config['D_ortho'] > 0.0:
                # Debug print to indicate we're using ortho reg in D.
                print('using modified ortho reg in D')
                utils.ortho(D, config['D_ortho'])

            D.optim.step()
            del D_loss

        # Optionally toggle "requires_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        ######################################
        # G-step
        ######################################
        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()
        encoder.module.optim.zero_grad()
        counter = 0

        # z_.sample_()
        # y_.sample_()
        # z_mean, z_log_var, z_ = encoder(x[counter])
        # z__ = torch.split(z_, split_size) #batch_size)
        # y__ = torch.split(y_, split_size) #batch_size)

        # If accumulating gradients, loop multiple times
        #** 2021-11-14 in this version change num_G_accumulations to 3 to enhance the similarity of input and synthesis 
        for accumulation_index in range(config['num_G_accumulations']): 

            G_fake, G_fake_middle, G_z, z_mean, z_log_var, z = GD( gy=y[counter],return_G_z=True, train_G=True, split_D=config['split_D'], reference_x = x[counter] )

            G_loss_fake_2d = fake_loss(G_fake) # BCE loss
            G_loss_fake_middle = fake_loss(G_fake_middle)# BCE loss
            ssim = ssim_package.SSIM().to("cuda")
            G_loss_ssim = - torch.log(ssim(G_z, x[counter]) + 1e-15)
            G_loss_l2 = torch.abs((G_z - x[counter])**2).mean()
            G_loss_l1 = torch.abs(G_z - x[counter]).mean()
            G_loss_lat = (z_log_var.exp() + z_mean ** 2 - 1 - z_log_var).mean()
            #** 2021-11-14 in this version enhance the ssim and l2 and l1's influence
            G_loss = 0.5*G_loss_fake_middle + 0.5*G_loss_fake_2d + G_loss_lat + 5*G_loss_ssim + G_loss_lat + 3*G_loss_l2  + 2*G_loss_l1
            # G_loss = 0.5*G_loss_fake_middle + 0.5*G_loss_fake_2d  + 3*G_loss_ssim + 2*G_loss_lat + 3*G_loss_l2  + 3*G_loss_l1

            G_loss = G_loss / float(config['num_G_accumulations'])


            G_loss_fake_middle_item = G_loss_fake_middle.detach().item()
            G_loss_fake_2d_item = G_loss_fake_2d.detach().item()
            G_loss_ssim_item = G_loss_ssim.detach().item()
            G_loss_l2_item = G_loss_l2.detach().item()
            G_loss_l1_item = G_loss_l1.detach().item()
            G_loss_item = G_loss.detach().item()
            if step_count%200 == 0:
                print("G_loss_total: %2f "%G_loss_item)
                print("G_l2: %2f"%G_loss_l2_item)
            G_loss.backward()
            counter += 1

        # Optionally apply modified ortho reg in G
        if config['G_ortho'] > 0.0:
            print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
            # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
            utils.ortho(G, config['G_ortho'],
                                    blacklist=[param for param in G.shared.parameters()])


        G.optim.step()
        encoder.module.optim.step()
        del G_loss

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
            ema.update(state_dict['itr'])


        # save intermediate losses
        if use_mixup_in_this_round and (config["consistency_loss"] or config["consistency_loss_and_augmentation"]) and config["num_D_steps"]>0:
            out["consistency"] = float(consistency_loss_item)

        out['G_loss'] = float(G_loss_item)
        if  not (config["full_batch_mixup"] and use_mixup_in_this_round) and config["num_D_steps"]>0:
            out['D_loss_real'] = float(D_loss_real_item)
            out['D_loss_fake'] = float(D_loss_fake_item)

        if use_mixup_in_this_round and not config["consistency_loss"] and config["num_D_steps"]>0:
            out["mixed_middle_loss"] = float(mixed_middle_loss_item)
            out["D_loss_mixed_2d"] = float(D_loss_mixed_2d_item)

        if  not (config["full_batch_mixup"] and use_mixup_in_this_round):
            if config["num_D_steps"]>0:
                out["D_loss_real_middle"] = float(D_loss_real_middle_item)
                out["D_loss_fake_middle"] = float(D_loss_fake_middle_item)
                out["D_loss_real_2d"] = float(D_loss_real_2d_item)
                out["D_loss_fake_2d"] = float(D_loss_fake_2d_item)
            out["G_loss_fake_middle"] = float(G_loss_fake_middle_item)
            out["G_loss_fake_2d"] = float(G_loss_fake_2d_item)

        writer.add_scalars('Losses', out, step_count)
        return out
    return train

''' This function takes in the model, saves the weights (multiple copies if
        requested), and prepares sample sheets: one consisting of samples given
        a fixed noise seed (to show how the model evolves throughout training),
        a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, encoder, G_ema, x, y,
                                        state_dict, config, experiment_name,sample_only=False, use_real = False, real_batch = None,
                                         id = "", mixed=False, target_map = None):
    if not sample_only:
        utils.save_weights(G, D, encoder, state_dict, config['weights_root'],
                                             experiment_name, None, G_ema if config['ema'] else None)
        # Save an additional copy to mitigate accidental corruption if process
        # is killed during a save (it's happened to me before -.-)
        if config['num_save_copies'] > 0:
            utils.save_weights(G, D, encoder, state_dict, config['weights_root'],
                                                 experiment_name,
                                                 'copy%d' %    state_dict['save_num'],
                                                 G_ema if config['ema'] else None)
            state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    else:
        # Use EMA G for samples or non-EMA?
        which_G = G_ema if config['ema'] and config['use_ema'] else G

        # # Accumulate standing statistics?
        # if config['accumulate_stats']:
        #     print("accumulating stats!")
        #     utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
        #                                                  z_, y_, config['n_classes'],
        #                                                  config['num_standing_accumulations'])

        # Save a random sample sheet with fixed z and y

        if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
            os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
        image_filename = '%s/%s/fixed_samples%d' % (config['samples_root'],experiment_name,state_dict['itr'])
        image_filename += id + ".png"

        if not (state_dict["itr"]>config["sample_every"] and use_real and not mixed):
            with torch.no_grad():
                reference_x = x.view(x.shape[0], 3, 8, 4, 8, 4)
                reference_x = reference_x.permute(0, 2, 1, 4, 3, 5)
                reference_x = reference_x.permute(0, 1, 3, 2, 4, 5)
                reference_x = reference_x.permute(1, 2, 0, 3, 4, 5)
                h_size, w_size, B, C, H, W = reference_x.shape
                l = [i for i in range(w_size*h_size)]
                random.shuffle(l)
                reference_x = reference_x.reshape(h_size*w_size, B, C, -1)
                # reference_x[l[0:21]] = 0 # mask ratio = 0.5
                reference_x = reference_x.permute(1, 2, 3, 0)
                reference_x = reference_x.reshape(B, C, H, W, h_size, w_size)
                reference_x = torch.cat([reference_x[..., i] for i in range(w_size)], 3)
                reference_x = torch.cat([reference_x[..., i] for i in range(h_size)], 2)
                erasing_transforms = transforms.RandomErasing(p=1, scale=(0.1, 0.33), ratio=(0.5,2.5))
                reference_x = erasing_transforms(x)
                if use_real:
                    fixed_Gz = real_batch
                    experiment_name += "_real"
                else:
                    if config['parallel']:
                        _, _, z = encoder(reference_x)
                        fixed_Gz =    nn.parallel.data_parallel(which_G, (z, which_G.shared(y)))
                        torchvision.utils.save_image(fixed_Gz[:64].float().cpu(), image_filename,
                                                            nrow=8, normalize=True)
                        del fixed_Gz
                        fixed_Gz = None
                        rint = torch.randint(0,99,(1,)).cuda()
                        y__ = torch.ones_like(y).cuda()
                        y__ = y__*rint
                        fixed_Gz_neg =    nn.parallel.data_parallel(which_G, (z, which_G.shared(y__)))
                        torchvision.utils.save_image(fixed_Gz_neg[:64].float().cpu(), image_filename[:-4]+"_"+str(rint.detach().item())+"_neg.png",
                                                                        nrow=8, normalize=True)
                        del fixed_Gz_neg
                        fixed_Gz_neg = None

                    else:
                        _, _, z = encoder(reference_x)
                        fixed_Gz = which_G(z, which_G.shared(y))
                        torchvision.utils.save_image(fixed_Gz[:64].float().cpu(), image_filename,
                                                            nrow=8, normalize=True)
                        del fixed_Gz
                        fixed_Gz = None
                        rint = torch.randint(0,99,(1,)).cuda()
                        y__ = torch.ones_like(y).cuda()
                        y__ = y__*rint
                        fixed_Gz_neg = which_G(z, which_G.shared(y__))
                        torchvision.utils.save_image(fixed_Gz_neg[:64].float().cpu(), image_filename[:-4]+"_"+str(rint.detach().item())+"_neg.png",
                                                                        nrow=8, normalize=True)
                        del fixed_Gz_neg
                        fixed_Gz_neg = None


                    torchvision.utils.save_image(reference_x[:64].float().cpu(), image_filename[:-4]+"_real.png",
                                                                    nrow=8, normalize=True)
                    del reference_x
                    reference_x = None



        # if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        #     os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
        # image_filename = '%s/%s/fixed_samples%d' % (config['samples_root'],experiment_name,state_dict['itr'])
        # image_filename += id + ".png"

        # if not (state_dict["itr"]>config["sample_every"] and use_real and not mixed):
        #     torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
        #                                                      nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)


        # with torch.no_grad():

        #     D_map, c = D(fixed_Gz ,y )
        #     D_map = F.sigmoid(D_map)
        #     c = F.sigmoid(c)

        #     s = D_map.mean([2,3])
        #     s = s.view(-1)
        #     c = c.view(-1)
        #     cs =  torch.cat((c.view(c.size(0),1)   ,s.view(s.size(0),1) ),dim=1)
        #     cs = cs.cpu().numpy()
        #     cs = cs.round(3)
        #     if mixed:
        #         s_real = D_map.clone()
        #         s_real = s_real*target_map # all fakes are zero now
        #         s_real = s_real.sum([2,3])/target_map.sum([2,3])

        #         s_fake = D_map.clone()
        #         s_fake = s_fake*(1-target_map) # all real are zero now
        #         s_fake = s_fake.sum([2,3])/(1-target_map).sum([2,3])

        #         s_fake = s_fake.view(-1)
        #         s_real = s_real.view(-1)

        #         cs_real =  torch.cat((c.view(c.size(0),1)   ,s_real.view(s_real.size(0),1) ),dim=1)
        #         cs_real = cs_real.cpu().numpy()
        #         cs_real = cs_real.round(3)

        #         cs_fake =  torch.cat((c.view(c.size(0),1)   ,s_fake.view(s_fake.size(0),1) ),dim=1)
        #         cs_fake = cs_fake.cpu().numpy()
        #         cs_fake = cs_fake.round(3)

        #         cs_mix =  torch.cat((c.view(c.size(0),1) ,s_real.view(s_real.size(0),1)  ,s_fake.view(s_fake.size(0),1) ),dim=1)
        #         cs_mix = cs_mix.cpu().numpy()
        #         cs_mix = cs_mix.round(3)

        #     for i in range(D_map.size(0)):
        #         D_map[i] = D_map[i] - D_map[i].min()
        #         D_map[i] = D_map[i]/D_map[i].max()

        #     image_filename = '%s/%s/fixed_samples_D%d' % (config['samples_root'],experiment_name,state_dict['itr'])
        #     image_filename += id + ".png"
            # import ipdb; ipdb.set_trace()
            # torchvision.utils.save_image(D_map.float().cpu(), image_filename,nrow=int(fixed_Gz.shape[0] **0.5), normalize=False)
            # import ipdb;ipdb.set_trace()

        # if config["resolution"]==128:
        #     num_per_sheet=16
        #     num_midpoints=8
        # elif config["resolution"]==256:
        #     num_per_sheet=8
        #     num_midpoints=4
        # elif config["resolution"]==64:
        #     num_per_sheet=32
        #     num_midpoints=16

        # if not use_real:
        #     # For now, every time we save, also save sample sheets
        #     utils.sample_sheet(which_G,
        #                                          classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
        #                                          num_classes=config['n_classes'],
        #                                          samples_per_class=10, parallel=config['parallel'],
        #                                          samples_root=config['samples_root'],
        #                                          experiment_name=experiment_name,
        #                                          folder_number=state_dict['itr'],
        #                                          z_=z_)


''' This function runs the inception metrics code, checks if the results
        are an improvement over the previous best (either in IS or FID,
        user-specified), logs the results, and saves a best_ copy if it's an
        improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
                 experiment_name, test_log, moments = "train"):
    print('Gathering inception metrics...')
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                                     z_, y_, config['n_classes'],
                                                     config['num_standing_accumulations'])
    IS_mean, IS_std, FID = get_inception_metrics(sample,config['num_inception_images'], num_splits=10)
    print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
    # If improved over previous best metric, save approrpiate copy
    if moments=="train":
        if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
            print('%s improved over previous best, saving checkpoint...' % config['which_best'])
            utils.save_weights(G, D, state_dict, config['weights_root'],
                                                 experiment_name, 'tr_best%d' % state_dict['save_best_num'],
                                                 G_ema if config['ema'] else None)
            state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
        state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
        state_dict['best_FID'] = min(state_dict['best_FID'], FID)
        # Log results to file
        test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
                                 IS_std=float(IS_std), FID=float(FID))
    elif moments=="test":
        if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS_test'])
            or (config['which_best'] == 'FID' and FID < state_dict['best_FID_test'])):
            print('%s improved over previous best, saving checkpoint...' % config['which_best'])
            utils.save_weights(G, D, state_dict, config['weights_root'],
                                                 experiment_name, 'te_best%d' % state_dict['save_best_num'],
                                                 G_ema if config['ema'] else None)

            state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
        state_dict['best_IS_test'] = max(state_dict['best_IS_test'], IS_mean)
        state_dict['best_FID_test'] = min(state_dict['best_FID_test'], FID)
        # Log results to file
        test_log.log(itr=int(state_dict['itr']), IS_mean_test=float(IS_mean),
                                 IS_std_test=float(IS_std), FID_test=float(FID))

    return IS_mean, IS_std, FID
