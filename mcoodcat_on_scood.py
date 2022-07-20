import os
import numpy as np
import argparse

use_tqdm=False
if use_tqdm:
    from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
####
from torchvision import transforms, utils
import misc.utils as utils
import os
import sys
import piq

# BLACKLIST = type, ModuleType, FunctionType
import matplotlib.pyplot as plt
import numpy
# from data_utils.load_dataset import LoadDatasets
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score

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

from typing import Dict 

import os
import torch
import numpy as np 
from tqdm import tqdm 
import logging
import sklearn.metrics as metrics

from torch.multiprocessing import Pool, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import os
import torch
import numpy as np
from tqdm import tqdm
import logging
import sklearn.metrics as metrics
from models.Dis_ood import Discriminator

sys.path.append('ICCV21_SCOOD')
from scood.data.utils import get_dataloader_self


def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))

def find_and_remove(path):
    if os.path.isfile(path):
        os.remove(path)

def constract_pair(clean_input, synthesis):
    # return clean_input - synthesis
    return torch.cat([clean_input, synthesis], dim=1)

_MULTI_PROCESS = False
_MULTI_PROCESS_INNER = True
_PRINT_AUX = False


def judge_thresh(l2_out, thresh, min_distance=False):
    # True for pass, False for reject
    if min_distance:
        return (l2_out < thresh).long()
    else:
        return (l2_out > thresh).long()

def _intialize_refine(x_item, y_item, thrs):
    x_refine, y_refine = [], []
    last_x = x_item[0]
    y_group = [y_item[0]]
    thrs_group = [thrs[0]]
    thrs_final, x_final = [], -1
    return {"x_refine": x_refine, "y_refine": y_refine, "last_x": last_x,
            "y_group": y_group, "thrs_group": thrs_group,
            "thrs_final": thrs_final, "x_final": x_final}


def _update_refine(x_item, y_item, thrs_sort, tpr_thrs, idx, direction,
        x_refine, y_refine, last_x, y_group, thrs_group, thrs_final, x_final):
    if direction(last_x, x_item[idx]):
        # 2a. accumulate same fpr
        y_group.append(y_item[idx])
        thrs_group.append(thrs_sort[idx])
    else:
        # 2b. determine best tpr
        y_group = np.array(y_group)
        tpr_max = np.max(y_group)
        tpr_max_ind = np.argmax(y_group)
        if len(y_refine) == 0 or tpr_max >= y_refine[-1]:
            x_refine.append(last_x)
            y_refine.append(tpr_max)
            # check if fpr reach drop rate
            if tpr_max >= tpr_thrs and len(thrs_final) == 0:
                thrs_final = [thrs_group[tpr_max_ind]]
                x_final = last_x
        elif _PRINT_AUX:
            print(f"drop point on {last_x}, {tpr_max}")
        y_group = [y_item[idx]]
        last_x = x_item[idx]
        thrs_group = [thrs_sort[idx]] 
    return {"x_refine": x_refine, "y_refine": y_refine, "last_x": last_x,
            "y_group": y_group, "thrs_group": thrs_group,
            "thrs_final": thrs_final, "x_final": x_final}


def _post_process(params):
    x_refine = params["x_refine"]
    y_refine = params["y_refine"]
    x_refine.append(params["last_x"])
    y_group = np.array(params["y_group"])
    y_refine.append(np.max(y_group))
    thrs_final = params["thrs_final"]
    x_final = params["x_final"]
    return x_refine, y_refine, thrs_final, x_final


def refine_inner_loop_double(fpr, tpr, prec, recall, i_prec, i_recall,
                             tf_thrs_sort, pr_thrs_sort, ipr_thrs_sort,
                             keep_rate):
    tf_param = _intialize_refine(fpr, tpr, tf_thrs_sort)
    pr_param = _intialize_refine(recall, prec, pr_thrs_sort)
    ipr_param = _intialize_refine(i_recall, i_prec, ipr_thrs_sort)
    for i in range(1, len(fpr)):
        tf_param = _update_refine(fpr, tpr, tf_thrs_sort, keep_rate, i, lambda x, y: x >= y, **tf_param)
        pr_param = _update_refine(recall, prec, pr_thrs_sort, keep_rate, i, lambda x, y: x <= y, **pr_param)
        ipr_param = _update_refine(i_recall, i_prec, ipr_thrs_sort, keep_rate, i, lambda x, y: x <= y, **ipr_param)
    fpr_refine, tpr_refine, tf_thrs_final, tpr_final = _post_process(tf_param)
    recall_refine, prec_refine, pr_thrs_final, recall_final = _post_process(pr_param)
    irecall_refine, iprec_refine, ipr_thrs_final, irecall_final = _post_process(ipr_param)
    return fpr_refine, tpr_refine, tf_thrs_final, tpr_final, \
        prec_refine, recall_refine, pr_thrs_final, recall_final, \
        iprec_refine, irecall_refine, ipr_thrs_final, irecall_final

    
# This is useless, just as a simple template for check.
def refine_inner_loop(fpr, tpr, prec, recall, tp_thrs_sort, pr_thrs_sort,
                      drop_rate):
    # 1. initialize
    fpr_refine, tpr_refine = [], []
    last_fpr = fpr[0]
    tpr_group = [tpr[0]]
    thresholds_group = [tp_thrs_sort[0]]
    thresholds_final, tpr_final = None, None
    for i in range(1, len(fpr)):
        if last_fpr >= fpr[i]:
            # 2a. accumulate same fpr
            tpr_group.append(tpr[i])
            thresholds_group.append(tp_thrs_sort[i])
        else:
            # 2b. determine best tpr
            tpr_group = np.array(tpr_group)
            tpr_max = np.max(tpr_group)
            tpr_max_ind = np.argmax(tpr_group)
            if len(tpr_refine) == 0 or tpr_max >= tpr_refine[-1]:
                fpr_refine.append(last_fpr)
                tpr_refine.append(tpr_max)
                # check if fpr reach drop rate
                if last_fpr <= drop_rate:
                    thresholds_final = thresholds_group[tpr_max_ind]
                    tpr_final = tpr_max
            tpr_group = [tpr[i]]
            last_fpr = fpr[i]
            thresholds_group = [tp_thrs_sort[i]]
    # 3. handle last group
    fpr_refine.append(last_fpr)
    tpr_group = np.array(tpr_group)
    tpr_refine.append(np.max(tpr_group))
    return fpr_refine, tpr_refine, thresholds_final, tpr_final


def refine_fpr_tpr(fpr, tpr, prec, recall, i_prec, i_recall, all_combinations,
                   keep_rate=0.95):
    """sort and check effective pair of fpr and tpr
    Args:
        fpr (List): points of fpr
        tpr (List): points of tpr
    Returns:
        fpr_refine, tpr_refine (List, List): fpr and tpr after refine
    """
    ######### check data ###############
    assert len(fpr) == len(tpr)
    assert len(prec) == len(recall)
    ######## process fpr, tpr #########
    # sort fpr
    fpr = torch.tensor(fpr).cuda()
    fpr, tf_inds = fpr.sort()
    fpr = fpr.cpu().numpy()
    tf_inds = tf_inds.cpu()
    # change order of thresholds and tpr accordingly
    tf_thrs_sort = torch.tensor(all_combinations)[tf_inds].numpy()
    tpr = torch.tensor(tpr)[tf_inds].numpy()
    ######## process prec, recall #########
    # sort recall
    recall = torch.tensor(recall).cuda()
    recall, pr_inds = recall.sort(descending=True)
    recall = recall.cpu().numpy()
    pr_inds = pr_inds.cpu()
    # change order of thresholds and tpr accordingly
    pr_thrs_sort = torch.tensor(all_combinations)[pr_inds].numpy()
    prec = torch.tensor(prec)[pr_inds].numpy()
    ######## process iprec, irecall #########
    # sort recall
    i_recall = torch.tensor(i_recall).cuda()
    i_recall, ipr_inds = i_recall.sort(descending=True)
    i_recall = i_recall.cpu().numpy()
    ipr_inds = ipr_inds.cpu()
    # change order of thresholds and tpr accordingly
    ipr_thrs_sort = torch.tensor(all_combinations)[ipr_inds].numpy()
    i_prec = torch.tensor(i_prec)[ipr_inds].numpy()
    ######### All data is now in numpy form
    return refine_inner_loop_double(fpr, tpr, prec, recall, i_prec, i_recall,
                                    tf_thrs_sort, pr_thrs_sort, ipr_thrs_sort,
                                    keep_rate)


@torch.no_grad()
def tpr_fpr_single_attack(
        min_dists, this_score, this_y, all_combinations, d_names, proc_id):
    """
    pass = 1 -> positive, label: pos -> 1, neg -> 0
    positive -> in-d, negative -> ood
    """
    ############# set device and move data to gpu ################
    gpu_id = proc_id % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    this_score = {k:v.cuda(gpu_id) for k,v in this_score.items()}
    this_y = this_y.cuda(gpu_id)
    all_combinations = torch.tensor(all_combinations).cuda(gpu_id)
    ##############################################################
    actualP = this_y.sum()
    actualN = len(this_y) - actualP
    fpr, tpr, prec, i_prec, i_recall = [], [], [], [], []
    for thresh in tqdm(all_combinations):
        all_pass = torch.ones_like(this_y)
        for idx, d_name in enumerate(d_names):
            this = judge_thresh(this_score[d_name], thresh[idx],
                                min_dists[d_name])
            all_pass = torch.logical_and(all_pass, 1 - this)
        # check fpr
        FP = torch.logical_and(this_y == 0, all_pass == 1).sum().item()
        # check tpr
        TP = torch.logical_and(this_y == 1, all_pass == 1).sum().item()
        ############ This is for inverse label ###########
        # we treat 1 -> ood, negative; 0 -> in-d, positive.
        iTP = torch.logical_and(this_y == 0, all_pass == 0).sum().item()
        ipredP = (all_pass == 0).sum().item()
        i_prec.append(iTP / float(ipredP) if ipredP > 0 else 1)
        iactualP = actualN # note the inverse
        i_recall.append(iTP / float(iactualP))
        ############ PAY ATTENTION #######################
        predP = (all_pass == 1).sum().item()
        fpr.append(FP / float(actualN))
        tpr.append(TP / float(actualP)) # this is also for recall.
        prec.append(TP / float(predP) if predP > 0 else 1)
    recall = tpr.copy()
    return fpr, tpr, prec, recall, i_prec, i_recall


def tpr_fpr_process(a_name, min_dists, this_score, this_y, all_combinations,
                    d_names, keep_rate):
    print("[{}] Start testing".format(a_name))
    num_processes = os.cpu_count()
    num_gpu = torch.cuda.device_count()
    num_spawn = 8 - 8 % num_gpu
    procs = []
    print(f"CPU count is {num_processes}")
    print(f"GPU count is {num_gpu}")
    print(f"spawn {num_spawn} process")
    actualP = this_y.sum()
    actualN = len(this_y) - actualP
    print("[{}] P={}, N={}".format(a_name, actualP, actualN))
    pool = Pool(num_spawn)
    block_size = int(np.ceil(len(all_combinations) / num_spawn))
    if _MULTI_PROCESS_INNER:
        for proci in range(num_spawn):
            procs.append(pool.apply_async(
                    tpr_fpr_single_attack,
                    args=(min_dists, this_score, this_y,
                          all_combinations[block_size*proci:block_size*(proci+1)],
                          d_names, proci)
                ))
        fpr, tpr, prec, recall, iprec, irecall = [], [], [], [], [], []
        for proc in procs:
            block_fpr, block_tpr, block_prec, block_recall, block_iprec, \
                block_irecall = proc.get()
            fpr += block_fpr
            tpr += block_tpr
            prec += block_prec
            recall += block_recall
            iprec += block_iprec
            irecall += block_irecall
    else:
       fpr, tpr, prec, recall, iprec, irecall = tpr_fpr_single_attack(
           min_dists, this_score, this_y, all_combinations, d_names, 0) 

    # fpr, tpr = tpr_fpr_single_attack(
    #     a_name, min_dists, this_score, this_y, all_combinations, d_names)
    # refine tpr and fpr
    fpr, tpr, tf_thresh, tpr_final,\
        prec, recall, pr_thresh, recall_final, \
        iprec, irecall, ipr_thresh, irecall_final = refine_fpr_tpr(
            fpr, tpr, prec, recall, iprec, irecall, all_combinations, keep_rate)
    # Q.put({a_name: {"fpr": fpr, "tpr": tpr, "final_thresh": final_thresh,
    #                  "tpr_final": tpr_final}})
    print(">>>>>>>>>> adding extreme values <<<<<<<<<<<<<")
    fpr = [0] + fpr + [1]
    tpr = [0] + tpr + [1]
    recall = [1] + recall + [0]
    prec = [0] + prec + [1]
    irecall = [1] + irecall + [0]
    iprec = [0] + iprec + [1]
    return {a_name: {"fpr": fpr, "tpr": tpr, "final_thresh": tf_thresh,
                     "fpr_final": tpr_final},
            a_name + "pr_curve": {"fpr": recall, "tpr": prec,
                                  "final_thresh": pr_thresh,
                                  "fpr_final": recall_final},
            a_name + "ipr_curve": {"fpr": irecall, "tpr": iprec,
                                   "final_thresh": ipr_thresh,
                                   "fpr_final": irecall_final}}


def check_above(x1, y1, x2, y2, xs, ys, inversed=False):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    after_refine_x = []
    after_refine_y = []
    for xi, yi in zip(xs, ys):
        D = A * xi + B * yi + C
        if (D <= 0 and not inversed) or (D >= 0 and inversed):
            after_refine_x.append(xi)
            after_refine_y.append(yi)
    return after_refine_x, after_refine_y


def pick_above_tpr(fpr1, tpr1, fpr2, tpr2, inversed=False):
    final_fpr = []
    final_tpr = []
    assert len(fpr2) >= len(fpr1)
    assert fpr2[0] == fpr1[0], "fprs should have common start point."
    idx2 = 0
    idx1 = 0
    if inversed:
        direct = lambda x, y: x < y
    else:
        direct = lambda x, y: x > y
    while True:
        accum_fpr = []
        accum_tpr = []
        if direct(fpr1[idx1], fpr2[idx2]):
            while idx2 < len(fpr2) and direct(fpr1[idx1], fpr2[idx2]):
                accum_fpr.append(fpr2[idx2])
                accum_tpr.append(tpr2[idx2])
                idx2 += 1
            temp_fpr, temp_tpr = check_above(
                fpr1[idx1 - 1], tpr1[idx1 - 1], fpr1[idx1], tpr1[idx1],
                accum_fpr, accum_tpr, inversed)
            final_fpr += temp_fpr
            final_tpr += temp_tpr
        elif fpr1[idx1] == fpr2[idx2]:
            final_fpr.append(fpr1[idx1])
            final_tpr.append(max(tpr1[idx1], tpr2[idx2]))
            idx1 += 1
            idx2 += 1
        else:
            while idx1 < len(fpr1) and direct(fpr2[idx2], fpr1[idx1]):
                accum_fpr.append(fpr1[idx1])
                accum_tpr.append(tpr1[idx1])
                idx1 += 1
            temp_fpr, temp_tpr = check_above(
                fpr2[idx2 - 1], tpr2[idx2 - 1], fpr2[idx2], tpr2[idx2],
                accum_fpr, accum_tpr, inversed)
            final_fpr += temp_fpr
            final_tpr += temp_tpr
        if idx1 >= len(fpr1):
            while idx2 < len(fpr2):
                final_fpr.append(fpr2[idx2])
                final_tpr.append(tpr2[idx2]) 
                idx2 += 1
            break
        if idx2 >= len(fpr2):
            while idx1 < len(fpr1):
                final_fpr.append(fpr1[idx1])
                final_tpr.append(tpr1[idx1]) 
                idx1 += 1
            break
    return final_fpr, final_tpr

# if __name__ == "__main__":
#     x1 = np.arange(0, 1.5, 0.05)
#     x1 = x1[::-1]
#     y1 = np.sin(x1)
#     x2 = np.array(np.arange(0, 0.7, 0.03).tolist() +
#                   np.arange(0.8, 1.5, 0.04).tolist() + x1[:1].tolist())
#     x2 = x2[::-1]
#     y2 = x2 ** 2
#     x3, y3 = pick_above_tpr(x1, y1, x2, y2, True)
#     plt.plot(x1, y1, label="sin")
#     plt.plot(x2, y2, label="x ** 2")
#     plt.plot(x3, y3, label="combine")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.xlim(0, 2)
#     plt.legend()
#     plt.savefig("temp.png")
#     exit(0)


def plot_roc(score: Dict[str, Dict[str, torch.Tensor]],
             y: Dict[str, torch.LongTensor],
             thresholds: Dict[str, torch.Tensor],
             save_name: str,
             keep_rate=0.95,
             prev_res=None):
    """plot roc curve.
    expect all clean samples can be classified correctly;
    expect all AE samples can attack successfully.
    Args:
        score (Dict[attack, Dict[d_name, score]]):
            dict of scores from each detector
        y (Dict[attack, torch.LongTensor]):
            1 for p/real AE sample; 0 for n/real clean sample
        thresholds (Dict[d_name, torch.Tensor]):
            dict of thresholds to each detecor
        model (ContraNetDict): detection dict model
        save_name (str): file to save
    """
    def merge(listA, listB):
        results = []
        for i in listA:
            for j in listB:
                if isinstance(i, list):
                    results.append(i + [j])
                else:
                    results.append([i, j])
        return results

    # get all combinations for thresholds
    d_names = list(thresholds.keys())
    all_combinations = [[i] for i in thresholds[d_names[0]].tolist()]
    for idx in range(len(d_names) - 1):
        all_combinations = merge(
            all_combinations, thresholds[d_names[idx + 1]].tolist())
    print("Total {} combinations to test".format(len(all_combinations)))
    # test for each attack
    min_dists = {d_n: True for d_n in d_names}

    all_auc, final_thresholds = [], []
    final_fpr = []
    results = {}
    for attack in score.keys():
        this_score = score[attack]
        this_y = y[attack]
        if _MULTI_PROCESS:
            raise NotImplementedError()
        else:
            # this_score = {k:v.cuda().share_memory_() for k,v in score[attack].items()}
            # this_y = this_y.cuda().share_memory_()
            # all_combinations = torch.tensor(all_combinations).cuda().share_memory_()
            this_res = tpr_fpr_process(attack, min_dists, this_score, this_y,
                                       all_combinations, d_names, keep_rate)
            results.update(this_res)
    if _MULTI_PROCESS:
        # for proc in procs:
        raise NotImplementedError()
        # results.update(Q.get())
    # set random color before draw
    if len(results.keys()) > 10:
        colormap = plt.cm.nipy_spectral  # nipy_spectral, Set1, Paired
    else:
        colormap = plt.get_cmap("tab10")  # defualt color
    for idx, attack in enumerate(results.keys()):
        tpr = results[attack]["tpr"]
        fpr = results[attack]["fpr"]
        print("[{}] {} points after refine".format(attack, len(fpr)))
        if prev_res is not None:
            for key, value in prev_res.items():
                prev_fpr, prev_tpr = value[attack]['fpr'], value[attack]['tpr']
                if idx > 0:
                    fpr, tpr = pick_above_tpr(prev_fpr, prev_tpr, fpr, tpr, True)
                else:
                    fpr, tpr = pick_above_tpr(prev_fpr, prev_tpr, fpr, tpr)
                print("[{}] {} points after refine on {}".format(attack, len(fpr), key))
        # print("fpr: ",fpr)
        # print("tpr: ",tpr)
        # for index, value in enumerate(tpr):
        #     if 0.96>=value>=0.949:
        #         print(f"tpr {value}, fpr:{fpr[index]}")
        fpr_final = results[attack]["fpr_final"]
        final_thresh = results[attack]["final_thresh"]
        print(f"[{attack}] thresh at keep rate {keep_rate}: " +
              f"{final_thresh[0].tolist()}, fpr={fpr_final:.4f}")
        roc_auc = metrics.auc(fpr, tpr)
        all_auc.append(roc_auc)
        final_thresholds += [final_thresh[0].tolist()]
        if idx == 0:
            final_fpr.append(fpr_final)
        print("[{}] roc_auc = {:.4f}".format(attack, roc_auc))
        print("[{}] Done".format(attack))
        color = colormap(idx / len(results.keys())) if \
            len(results.keys()) > 10 else colormap(idx)
        plt.plot(fpr, tpr, label="{} auc={:.4f}".format(
            attack, roc_auc), color=color)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(os.path.basename(save_name).split(".")[0])
    plt.legend()
    plt.savefig(save_name)
    plt.close()
    return all_auc, final_thresholds, final_fpr, results


def fix_all_seed(seed):
    print(f"********** set seed to {seed} ***********")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_deterministic(True)


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
    SEED = 13344
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
    
    encoder.cuda()
    G.cuda()
    Dis.cuda()
    pure_Dis.cuda()
    # encoder = nn.DataParallel(encoder)
    if config['parallel']:
        encoder = nn.DataParallel(encoder)
        G = nn.DataParallel(G)
        Dis = nn.DataParallel(Dis)
        pure_Dis = nn.DataParallel(pure_Dis)

    # define detector 
    # content_loss = piq.ContentLoss(
        # feature_extractor="vgg16", layers=('relu3_3',), reduction='none')
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


    pass_rate = 0.95
    clf_name = 'DISTS+SSIM+LPIPS'

    print(f"=========={OOD} with {DETECTOR_CHOICE} on bench {BENCHMARK}==============")
    
    ood_name = ood_map[OOD]

    # data_transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), ( 0.5, 0.5, 0.5))])
    in_data_loader = get_dataloader_self(benchmark=BENCHMARK, name=BENCHMARK, batch_size=batch_size, num_classes=CLASS_NUM)
    out_data_loader = get_dataloader_self(benchmark=BENCHMARK, name=ood_name, batch_size=batch_size, num_classes=CLASS_NUM)

    encoder.eval()
    G.eval()
    erasing_transforms = transforms.RandomErasing(p=1, scale=(0.15, 0.33), ratio=(0.5,2.5))
    if config["progress_bar"]: # * True
        if config['pbar'] == 'mine':
            pbar = utils.progress(in_data_loader, displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(in_data_loader)
    else:
        pbar = in_data_loader


    # in_data_test = iter(in_data_test_loader)
    fix_all_seed(SEED)
    for data_id, batch_data in enumerate(pbar):# * the i indicates one batch data
        x = batch_data["data"].cuda()
        y = batch_data["label"].cuda()
        input_x = erasing_transforms(x)
        _, _, z = encoder(input_x.cuda())# * former half contains in_dis, latter half standing for OOD.
        assert (y < CLASS_NUM).all()
        synthesis = G(z,G.module.shared(y))
        # negative_pair stands for clean samples 0 for clean 
        negative_pair_piq_score_1 =  piq_detector_1((x+1)/2., (synthesis+1)/2.).cpu() 
        negative_pair_piq_score_2 =  piq_detector_2((x+1)/2., (synthesis+1)/2.).cpu() 
        negative_pair_piq_score_3 =  piq_detector_3((x+1)/2., (synthesis+1)/2.).cpu() 
        # import ipdb; ipdb.set_trace()
        # negative_pair_piq_score_4 =  piq_detector_4(re((x+1)/2.), re((synthesis[:batch_size]+1)/2.), as_loss=False).detach().cpu() 
        negative_pair_piq_score_4 =  piq_detector_4((x+1)/2., (synthesis+1)/2., as_loss=False ).cpu() #

        negative_pair_piq_score_5 =  piq_detector_5((x+1)/2., (synthesis+1)/2.).cpu() 
        
        negative_pair_piq_score_7 =  piq_detector_7((synthesis+1)/2.).cpu() 
        negative_pair = constract_pair(x, synthesis)
        negative_pair_piq_score_8 = piq_detector_8(negative_pair, y).cpu()
        negative_pair_piq_score_9 = piq_detector_9(negative_pair, y).cpu()
        if data_id == 0:
            negative_tank_1 = negative_pair_piq_score_1 # * the former part is ood test data, the latter part is in-dist test data (1, 0)
            negative_tank_2 = negative_pair_piq_score_2 # * the former part is ood test data, the latter part is in-dist test data (1, 0)
            negative_tank_3 = negative_pair_piq_score_3 # * the former part is ood test data, the latter part is in-dist test data (1, 0)
            negative_tank_4 = negative_pair_piq_score_4 # * the former part is ood test data, the latter part is in-dist test data (1, 0)
            negative_tank_5 = negative_pair_piq_score_5 # * the former part is ood test data, the latter part is in-dist test data (1, 0)
            negative_tank_7 = negative_pair_piq_score_7 # * the former part is ood test data, the latter part is in-dist test data (1, 0)
            negative_tank_8 = negative_pair_piq_score_8
            negative_tank_9 = negative_pair_piq_score_9
        else:
            negative_tank_1 = torch.cat([negative_tank_1, negative_pair_piq_score_1])
            negative_tank_2 = torch.cat([negative_tank_2, negative_pair_piq_score_2])
            negative_tank_3 = torch.cat([negative_tank_3, negative_pair_piq_score_3])
            negative_tank_4 = torch.cat([negative_tank_4, negative_pair_piq_score_4])
            negative_tank_5 = torch.cat([negative_tank_5, negative_pair_piq_score_5])
            negative_tank_7 = torch.cat([negative_tank_7, negative_pair_piq_score_7])
            negative_tank_8 = torch.cat([negative_tank_8, negative_pair_piq_score_8])
            negative_tank_9 = torch.cat([negative_tank_9, negative_pair_piq_score_9])
            
    if config["progress_bar"]: # * True
        if config['pbar'] == 'mine':
            pbar = utils.progress(out_data_loader, displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
        else:
            pbar = tqdm(out_data_loader)
    else:
        pbar = out_data_loader

    fix_all_seed(SEED)
    for data_id, batch_data in enumerate(pbar):# * the i indicates one batch data
        x_out = batch_data["data"].cuda()
        y_out = batch_data["label"].cuda()
        input_x = erasing_transforms(x_out)
        _, _, z = encoder(input_x.cuda())# * former half contains in_dis, latter half standing for OOD.
        y_randn = torch.randint_like(y_out, 0, CLASS_NUM) # generate [0, 9]
        assert (y_randn < CLASS_NUM).all()
        synthesis = G(z,G.module.shared(y_randn))

        positive_pair_piq_score_1 =  piq_detector_1((x_out+1)/2, (synthesis+1)/2.).cpu()
        positive_pair_piq_score_2 =  piq_detector_2((x_out+1)/2, (synthesis+1)/2.).cpu()
        positive_pair_piq_score_3 =  piq_detector_3((x_out+1)/2, (synthesis+1)/2.).cpu()
        # positive_pair_piq_score_4 =  piq_detector_4(re((x_out+1)/2), re((synthesis[batch_size:]+1)/2.), as_loss=False).detach().cpu()
        positive_pair_piq_score_4 =  piq_detector_4((x_out+1)/2, (synthesis+1)/2., as_loss=False ).cpu()#, as_loss=False
        
        positive_pair_piq_score_5 =  piq_detector_5((x_out+1)/2, (synthesis+1)/2.).cpu()


        positive_pair_piq_score_7 =  piq_detector_7((synthesis+1)/2.).cpu()
        positive_pair = constract_pair(x_out, synthesis)       
        positive_pair_piq_score_8 = piq_detector_8(positive_pair, y_randn).cpu()    
        positive_pair_piq_score_9 = piq_detector_9(positive_pair, y_randn).cpu()    

        if data_id == 0:
            positive_tank_1 = positive_pair_piq_score_1
            positive_tank_2 = positive_pair_piq_score_2
            positive_tank_3 = positive_pair_piq_score_3
            positive_tank_4 = positive_pair_piq_score_4
            positive_tank_5 = positive_pair_piq_score_5
            positive_tank_7 = positive_pair_piq_score_7
            positive_tank_8 = positive_pair_piq_score_8 
            positive_tank_9 = positive_pair_piq_score_9 
        else:
            positive_tank_1 = torch.cat([positive_tank_1, positive_pair_piq_score_1])
            positive_tank_2 = torch.cat([positive_tank_2, positive_pair_piq_score_2])
            positive_tank_3 = torch.cat([positive_tank_3, positive_pair_piq_score_3])
            positive_tank_4 = torch.cat([positive_tank_4, positive_pair_piq_score_4])
            positive_tank_5 = torch.cat([positive_tank_5, positive_pair_piq_score_5])
            positive_tank_7 = torch.cat([positive_tank_7, positive_pair_piq_score_7])
            positive_tank_8 = torch.cat([positive_tank_8, positive_pair_piq_score_8])
            positive_tank_9 = torch.cat([positive_tank_9, positive_pair_piq_score_9])
        
    del G, encoder, piq_detector_8, Dis, pure_Dis
    return positive_tank_1, positive_tank_2, positive_tank_3, positive_tank_4, \
         positive_tank_5, positive_tank_7, positive_tank_8, negative_tank_1, \
         negative_tank_2, negative_tank_3, negative_tank_4, negative_tank_5, \
         negative_tank_7, negative_tank_8, DETECTOR_CHOICE, OOD, pass_rate, \
         args, positive_tank_9, negative_tank_9


def eval_auc(positive_tank_1, positive_tank_2, positive_tank_3, positive_tank_4,
             positive_tank_5, positive_tank_7, positive_tank_8, negative_tank_1,
             negative_tank_2, negative_tank_3, negative_tank_4, negative_tank_5,
             negative_tank_7, negative_tank_8, DETECTOR_CHOICE, OOD, pass_rate,
             args, positive_tank_9, negative_tank_9):
    print("positive length %d" % len(positive_tank_1), "negative length %d" % len(negative_tank_1))

    positive_tank_1 = positive_tank_1.numpy()
    positive_tank_2 = positive_tank_2.numpy()
    positive_tank_3 = positive_tank_3.numpy()
    positive_tank_4 = positive_tank_4.numpy()
    positive_tank_5 = positive_tank_5.numpy()
    positive_tank_7 = positive_tank_7.numpy()
    positive_tank_8 = positive_tank_8.numpy()
    positive_tank_9 = positive_tank_9.numpy()



    negative_tank_1 = negative_tank_1.numpy()
    negative_tank_2 = negative_tank_2.numpy()
    negative_tank_3 = negative_tank_3.numpy()
    negative_tank_4 = negative_tank_4.numpy()
    negative_tank_5 = negative_tank_5.numpy()
    negative_tank_7 = negative_tank_7.numpy()
    negative_tank_8 = negative_tank_8.numpy()
    negative_tank_9 = negative_tank_9.numpy()


    x_train_1 = -np.concatenate((positive_tank_1, negative_tank_1), axis=0)
    x_train_2 = np.concatenate((positive_tank_2, negative_tank_2), axis=0)
    x_train_3 = np.concatenate((positive_tank_3, negative_tank_3), axis=0)
    x_train_4 = -np.concatenate((positive_tank_4, negative_tank_4), axis=0)
    x_train_5 = -np.concatenate((positive_tank_5, negative_tank_5), axis=0)
    x_train_7 = np.concatenate((positive_tank_7, negative_tank_7), axis=0)
    x_train_8 = -np.concatenate((positive_tank_8, negative_tank_8), axis=0)
    x_train_9 = -np.concatenate((positive_tank_9, negative_tank_9), axis=0)

    if True:
        x_train_1 = (x_train_1-x_train_1.min())/(x_train_1.max() - x_train_1.min())
        x_train_1_cut = np.round(x_train_1, decimals=3)
        print(x_train_1.max(), x_train_1.min())

        x_train_2 = (x_train_2-x_train_2.min())/(x_train_2.max() - x_train_2.min())
        x_train_2_cut = np.round(x_train_2, decimals=3)
        print(x_train_2.max(), x_train_2.min())

        x_train_3 = (x_train_3-x_train_3.min())/(x_train_3.max() - x_train_3.min())
        x_train_3_cut = np.round(x_train_3, decimals=3)
        print(x_train_3.max(), x_train_3.min())
        
        # x_train_4 = np.around((x_train_4-x_train_4.mean())/x_train_4.var(), decimals=3)
        # print(x_train_4.max(), x_train_4.min())
        
        x_train_5 = (x_train_5-x_train_5.min())/(x_train_5.max() - x_train_5.min())
        x_train_5_cut = np.round(x_train_5, decimals=3)
        print(x_train_5.max(), x_train_5.min())
        
        # x_train_6 = np.around((x_train_6-x_train_6.mean())/x_train_6.var(), decimals=1)
        # print(x_train_6.max(), x_train_6.min())
        
        x_train_8 = (x_train_8-x_train_8.min())/(x_train_8.max() - x_train_8.min())
        x_train_8_cut = np.round(x_train_8, decimals=3)
        print(x_train_8.max(), x_train_8.min())
        
        x_train_9 = (x_train_9-x_train_9.min())/(x_train_9.max() - x_train_9.min())
        x_train_9_cut = np.round(x_train_9, decimals=3)
        print(x_train_9.max(), x_train_9.min())

    # * y_train is the groundtruth
    y_train = np.concatenate((np.zeros((len(positive_tank_1),), dtype=int), np.ones((len(negative_tank_1),), dtype=int)), axis=0) # * in_dis are zeros, ood are ones
    # y_train = torch.cat([torch.ones(len(positive_tank_1)), torch.zeros(len(negative_tank_1))]) # oods are ones, in_dis are zeros


    fpr_1, tpr_1, thresholds_1 = roc_curve(y_train, x_train_1_cut)
    print("auc of detector 1:",roc_auc_score(y_train, x_train_1))
    print("length of thresh 1:",len(thresholds_1))

    fpr_2, tpr_2, thresholds_2 = roc_curve(y_train, x_train_2_cut)
    print("auc of detector 2:",roc_auc_score(y_train, x_train_2))
    print("length of thresh 2:",len(thresholds_2))


    fpr_3, tpr_3, thresholds_3 = roc_curve(y_train, -x_train_3_cut)
    print("auc of detector 3:",roc_auc_score(y_train, -x_train_3))
    print("length of thresh 3:",len(thresholds_3))
    
    # fpr_4, tpr_4, thresholds_4 = roc_curve(y_train, x_train_4_cut)
    # print("auc of detector 4:",roc_auc_score(y_train, x_train_4))
    # print("length of thresh 4:",len(thresholds_4))

    fpr_5, tpr_5, thresholds_5 = roc_curve(y_train, x_train_5_cut)
    print("auc of detector 5:",roc_auc_score(y_train, x_train_5))
    print("length of thresh 5:",len(thresholds_5))

    # fpr, tpr, thresholds_6 = roc_curve(y_train, x_train_6)
    # print("auc of detector 6:",roc_auc_score(y_train, x_train_6))
    # print("length of thresh 6:",len(thresholds_6))

    fpr_8, tpr_8, thresholds_8 = roc_curve(y_train, x_train_8_cut)
    print("auc of detector 8:",roc_auc_score(y_train, x_train_8))
    print("length of thresh 8:",len(thresholds_8))

    fpr_9, tpr_9, thresholds_9 = roc_curve(y_train, x_train_9_cut)
    print("auc of detector 9:",roc_auc_score(y_train, x_train_9))
    print("length of thresh 9:",len(thresholds_9))
    # exit()
    # fpr_list = [fpr_1, fpr_4, fpr_5, fpr_8]
    # thrs_list = [thresholds_1, thresholds_4, thresholds_5, thresholds_8]
    # tpr_list = [tpr_1, tpr_4, tpr_5, tpr_8]
    # all_fpr = np.concatenate(fpr_list, axis=0)
    # all_tpr = np.concatenate(tpr_list, axis=0)
    # all_combinations = np.zeros([len(all_fpr), len(fpt_list)], dtype=np.float32)
    # current_len = 0
    # for d_num, (this_fpr, this_thrs) in enumerate(zip(fpr_list, thrs_list)):
    #     all_combinations[current_len:current_len + len(this_fpr), d_num] = this_thrs
    #     current_len += len(this_fpr)
    # fpr_refine, tpr_refine, thresholds_final, tpr_final = refine_fpr_tpr(all_fpr, all_tpr, all_combinations)
    # roc_auc = metrics.auc(fpr_refine, tpr_refine)
    # print("roc_auc:", roc_auc)
    # print("final thresholds:", thresholds_final)
    # print("tpr_final:", tpr_final)

    # exit()
    # cut_points = np.arange(0, 0.1, 0.005).tolist() + \
    #     np.arange(0.1, 0.3, 0.01).tolist() + \
    #         np.arange(0.3, 0.7, 0.05).tolist() + \
    #             np.arange(0.7, 0.9, 0.01).tolist() + \
    #                 np.arange(0.9, 1.00001, 0.005).tolist()
    # # cut_points = np.arange(0, 1.0005, 0.1)
    # thresholds_1 = x_train_1
    # thresholds_4 = x_train_4
    # thresholds_5 = x_train_5
    # thresholds_8 = x_train_8
    # thresholds_9 = x_train_9
    # inds_1 = (torch.tensor(cut_points)*(len(thresholds_1) - 1)).long()
    # inds_4 = (torch.tensor(cut_points)*(len(thresholds_4) - 1)).long()
    # inds_5 = (torch.tensor(cut_points)*(len(thresholds_5) - 1)).long()
    # inds_8 = (torch.tensor(cut_points)*(len(thresholds_8) - 1)).long()
    # inds_9 = (torch.tensor(cut_points)*(len(thresholds_9) - 1)).long()

    
    # thresholds_1 = torch.from_numpy(thresholds_1).sort()[0][inds_1]
    # thresholds_4 = torch.from_numpy(thresholds_4).sort()[0][inds_4]
    # thresholds_5 = torch.from_numpy(thresholds_5).sort()[0][inds_5]
    # thresholds_8 = torch.from_numpy(thresholds_8).sort()[0][inds_8]
    # thresholds_9 = torch.from_numpy(thresholds_9).sort()[0][inds_9]


    all_results_dict = {}
    for THIS_CHOICE in DETECTOR_CHOICE:
        score_dict = {}
        detector_dict = {}
        y_dict = {}
        thresholds = {}
        prev_res = None
        print(f"===={OOD} with {THIS_CHOICE} on bench {args.ind_dataset }====")
        THIS_CHOICE = str(THIS_CHOICE)
        for di in THIS_CHOICE:
            detector_dict[f'{di}'] = eval(f"torch.from_numpy(x_train_{di})")
            print(f"Add detector {di} in to detector dict.")
        score_dict[OOD] = detector_dict
        y_dict[OOD] = torch.from_numpy(y_train) 

        for idx, di in enumerate(THIS_CHOICE):
            thresholds[f'{di}'] = eval(f"thresholds_{di}")
            print(f"Add detector {di} in to evaluation thresholds.")
        if len(THIS_CHOICE) > 1:
            prev_res = {}
            for idx, di in enumerate(THIS_CHOICE):
                prev_res[f'{di}'] = all_results_dict[di]


        # thresholds['1'] = thresholds_1
        # thresholds['2'] = thresholds_2
        # thresholds['3'] = thresholds_3
        # thresholds['4'] = thresholds_4
        # thresholds['5'] = thresholds_5
        # thresholds['6'] = thresholds_6
        # thresholds['7'] = thresholds_7
        # thresholds['8'] = thresholds_8


        # tag_str = str().join([str(i) for i in DETECTOR_CHOICE])
        tag_str = THIS_CHOICE
        dirname = f"results/roc_curve_scood/{args.ind_dataset}_{tag_str}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        save_name = os.path.join(dirname, f"cascade_piq_detector_roc_curve_{OOD}_{tag_str}.pdf")
        # import ipdb; ipdb.set_trace()
        all_auc, final_thresholds, fpr_final, results_dict = plot_roc(
            score_dict, y_dict, thresholds, save_name, keep_rate=pass_rate,
            prev_res = prev_res
        )
        all_results_dict[THIS_CHOICE] = results_dict
        print(f"all auc: {all_auc}")
        print(f"final_thresholds: {final_thresholds}")
        csv_file = os.path.join(os.path.dirname(dirname), "results.csv")
        if not os.path.exists(csv_file):
            with open(csv_file, 'w') as f:
                f.write("ind,ood,detector,fpr,auc,auprin,auprout\n")
        with open(csv_file, 'a') as f:
            line = args.ind_dataset + "," + args.ood_dataset + "," + tag_str + ","
            line += f"{fpr_final[0]:.4f},"
            line += ",".join([f"{i:.4f}" for i in all_auc])
            line += "\n"
            f.write(line)

    # around_thresholds_1 = np.around(thresholds_1, decimals=3)
    # around_thresholds_2 = np.around
    return


@torch.no_grad()
def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_known_args()[0])
    if config["gpus"] !="":
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    keys = sorted(config.keys())

    positive_tank_1, positive_tank_2, positive_tank_3, positive_tank_4, \
    positive_tank_5, positive_tank_7, positive_tank_8, negative_tank_1, \
    negative_tank_2, negative_tank_3, negative_tank_4, negative_tank_5, \
    negative_tank_7, negative_tank_8, DETECTOR_CHOICE, OOD, pass_rate, args, \
    positive_tank_9, negative_tank_9 = run(config)
    eval_auc(positive_tank_1, positive_tank_2, positive_tank_3, positive_tank_4,
             positive_tank_5, positive_tank_7, positive_tank_8, negative_tank_1,
             negative_tank_2, negative_tank_3, negative_tank_4, negative_tank_5,
             negative_tank_7, negative_tank_8, DETECTOR_CHOICE, OOD, pass_rate,
             args, positive_tank_9, negative_tank_9)


if __name__ == '__main__':
    main()
