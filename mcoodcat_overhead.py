import os
import numpy as np
import argparse

use_tqdm=False
if use_tqdm:
    from tqdm import tqdm, trange
import torch
import torch.nn as nn
####
from torchvision import utils
import misc.utils as utils
import os
import sys
import piq

import numpy

import random
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)

from rich import print
import numpy as np 
import matplotlib as mpl 
mpl.use("Agg")
import numpy
from IQA_pytorch import NLPD, CW_SSIM, MAD, LPIPSvgg, SteerPyrComplex


import os
import torch
import numpy as np 
from tqdm import tqdm 
import logging

from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import os
import torch
import numpy as np
from tqdm import tqdm
import logging
from models.Dis_ood import Discriminator

from models.layers import SNConv2d, SNLinear
from models.model_ops import Self_Attn
from thop.vision.counter import counter_softmax

def count_self_attn(m, x, y):
    x = x[0]
    B, ch, h, w = x.size()
    theta_ch = ch // 8
    theta_size = h*w
    phi_size = h * w // 4
    total_ops = theta_ch ** 2 * theta_size * phi_size
    g_size = h * w // 4
    g_ch = ch // 2
    total_ops += g_size ** 2 * g_ch * h * w
    total_ops += counter_softmax(1, phi_size)
    total_ops += g_ch * h * w
    m.total_ops += total_ops * B


from thop import profile
from thop import clever_format
from thop.profile import register_hooks
register_hooks[SNConv2d] = register_hooks[nn.Conv2d]
register_hooks[SNLinear] = register_hooks[nn.Linear]
register_hooks[Self_Attn] = count_self_attn

sys.path.append('ICCV21_SCOOD')
from scood.data.utils import get_dataloader_self

from pytorch_cifar.models.resnet import ResNet18, ResNet34, ResNet50


@torch.no_grad()
def run(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ind_dataset", type=str, default="cifar10")
    parser.add_argument("--ood_dataset", type=str, default="CIFAR100")
    parser.add_argument("--test_detector", type=int, nargs="+", default=[1, 4, 5, 8])
    args = parser.parse_known_args()[0]
    print(args.test_detector)

    OOD = args.ood_dataset
    DETECTOR_CHOICE = args.test_detector
    BENCHMARK = args.ind_dataset
    batch_size = config['batch_size']
    if args.ind_dataset == "cifar10":
        CLASS_NUM = 10
        G_load_path = "ckpt/cifar10_ind/G_ep_3999.pth" #94.408
        E_load_path = "ckpt/cifar10_ind/E_ep_3999.pth" #94.408
        D_load_path = 'ckpt/cifar10_ind/discriminator_cat6_checkpoints/acc=D-best-weights-step=6500acc=0.8583734035491943.pth'
        pure_D_load_path = "ckpt/cifar10_ind/discriminator_cat6_ckpt_with_IND_cifar10/acc=D-best-weights-step=7930acc=0.8091946840286255.pth"
        ood_map = {"CIFAR100": "cifar100", "LSUN": "lsun", "Places365": "places365",
                   "SVHN": "svhn", "Texture": "texture", "Tiny_imagenet": "tin"}
    elif args.ind_dataset == "cifar100":
        CLASS_NUM = 100
        G_load_path = "ckpt/cifar100_ind/G_ep_3507.pth"
        E_load_path = "ckpt/cifar100_ind/E_ep_3507.pth"
        D_load_path = 'ckpt/cifar100_ind/discriminator_cat6_ckpt_cifar100/acc=D-best-weights-step=7930acc=0.8181423544883728.pth'
        pure_D_load_path = "ckpt/cifar100_ind/discriminator_cat6_ckpt_with_IND_cifar100/acc=D-best-weights-step=23920acc=0.7821180820465088.pth"
        ood_map = {"CIFAR10": "cifar10", "LSUN": "lsun", "Places365": "places365",
                   "SVHN": "svhn", "Texture": "texture", "Tiny_imagenet": "tin"}
    else:
        raise NotImplementedError()

    # logging config
    # create logger
    logger = logging.getLogger('piq_detector')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s  - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


    config['resolution'] = 32 #utils.imsize_dict[config['dataset']]
    config['n_classes'] = CLASS_NUM #utils.nclass_dict[config['dataset']]
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    # Import the model--this line allows us to dynamically select different files.
    model_sel = __import__(f"models.{config['model']}")
    model = eval(f"model_sel.{config['model']}")
    
    encoder = model.Encoder(isize=32, nz=128, nc=3, ndf=32)
    G = model.Generator(**config)
    Dis = Discriminator(32, 32, True, True, 1,
          "ReLU","ProjGAN" , "N/A", CLASS_NUM, False,
          False, "ortho", "N/A", False, input_dim=6)
    pure_Dis = Discriminator(32, 32, True, True, 1,
          "ReLU","ProjGAN" , "N/A", CLASS_NUM, False,
          False, "ortho", "N/A", False, input_dim=6)


    D_checkpoint = torch.load(D_load_path, map_location="cpu")
    Dis.load_state_dict(D_checkpoint['state_dict'])
    print(f"D loaded from {D_load_path}")

    pure_D_checkpoint = torch.load(pure_D_load_path, map_location="cpu")
    pure_Dis.load_state_dict(pure_D_checkpoint['state_dict'])
    print(f"D loaded from {pure_D_load_path}")

    G_checkpoint = torch.load(G_load_path, map_location="cpu")
    E_checkpoint = torch.load(E_load_path, map_location="cpu")
    G.load_state_dict(G_checkpoint)
    print(f"loaded G mdoel from {G_load_path}")
    encoder.load_state_dict(E_checkpoint)
    print(f"loaded encoder model from {E_load_path}")
    
    dists = piq.DISTS(reduction='none') #* threshold: -0.26738083 fpr=0.03
    dss = piq.DSSLoss(data_range=1., reduction='none')
    fsim = piq.FSIMLoss(data_range=1., reduction='none')#, scales=8,min_length=6)
    gmsd= piq.GMSDLoss(data_range=1., reduction='none')
    haarpsi = piq.HaarPSILoss(data_range=1., reduction='none')
    mdsi = piq.MDSILoss(data_range=1., reduction='none', c1=10)
    ms_ssim = piq.MultiScaleSSIMLoss(data_range=1., reduction='none',scale_weights = torch.tensor([0.0448, 0.2856, 0.3001]), kernel_size=7)
    tv = piq.TVLoss( reduction='none', norm_type='l1')
    # ms_ssim = piq.MultiScaleSSIMLoss(data_range=1., reduction='none')

    ms_gmsd = piq.MultiScaleGMSDLoss(chromatic=True, data_range=1., reduction='none')
    pieapp =  piq.PieAPP(reduction='none', stride=32)
    style = piq.StyleLoss(feature_extractor="vgg16", layers=("relu3_3",), reduction='none')
    vif = piq.VIFLoss(sigma_n_sq=2.0, data_range=1., reduction='none')
    vsi = piq.VSILoss(data_range=1., reduction='none')
    ssim = piq.SSIMLoss(data_range=1., reduction='none', kernel_size=21) #* threshold: -0.26956725 fpr=1%
    srsim = piq.SRSIMLoss(data_range=1., reduction='none',gaussian_size=7, chromatic=True)
    lpips = piq.LPIPS(reduction='none')
    brisque = piq.BRISQUELoss(reduction='none', data_range=1.)

    nlpd = NLPD(channels=3).cuda()
    lpipsvgg = LPIPSvgg(channels=3).cuda()
    mad = MAD(channels=3).cuda()
    # vifq = VIF(channels=3,imgSize=[32,32]).cuda()
    cw_ssim = CW_SSIM(channels=3, imgSize=[32,32],level=3,ori=4).cuda()
    spc = SteerPyrComplex.SteerablePyramid(imgSize=[32,32]).cuda()
    piq_detector_1 = dists
    piq_detector_2 = srsim
    piq_detector_3 = mdsi
    piq_detector_4 = mad #spc#vifq#cw_ssim
    piq_detector_5 = lpips
    piq_detector_7 = brisque
    piq_detector_8 = Dis
    piq_detector_9 = pure_Dis
    
    fake_y = torch.zeros(1, dtype=torch.long)
    fake_x = torch.randn(1, 3, 32, 32)
    print(f"========== for Encoder ==============")
    macs, params = profile(encoder, inputs=(fake_x,))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params)

    print(f"========== for Generator ==============")
    fake_z = torch.randn(1, 128)
    # print(G)
    macs, params = profile(G, inputs=(fake_z, G.shared(fake_y)))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params)

    print(f"========== for Discriminator ==============")
    fake_x_cat = torch.randn(1, 6, 32, 32)
    # print(Dis)
    macs, params = profile(Dis, inputs=(fake_x_cat, fake_y))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params)

    print(f"========== for 1: dists ==============")
    macs, params = profile(piq_detector_1, inputs=(fake_x, fake_x))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params) 

    print(f"========== for 5: lpips ==============")
    macs, params = profile(piq_detector_5, inputs=(fake_x, fake_x))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params) 

    print(f"========== for classifier ResNet18 ==============")
    clssifier = ResNet18()
    macs, params = profile(clssifier, inputs=(fake_x, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params) 

    print(f"========== for classifier ResNet34 ==============")
    clssifier = ResNet34()
    macs, params = profile(clssifier, inputs=(fake_x, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params)

    print(f"========== for classifier ResNet50 ==============")
    clssifier = ResNet50()
    macs, params = profile(clssifier, inputs=(fake_x, ))
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:' + macs)
    print('Params:' + params) 

    exit(0)

@torch.no_grad()
def main():

    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_known_args()[0])
    if config["gpus"] !="":
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    keys = sorted(config.keys())

    run(config)

if __name__ == '__main__':
    main()