#!/bin/bash

set -x

export CUBLAS_WORKSPACE_CONFIG=:16:8

PARAMS="--dataset CIFAR10 --parallel --batch_size 384  --which_best FID \
--num_D_steps 1  --G_lr 5e-5  --D_lr 2e-4  \
--D_B2 0.999 --G_B2 0.999 --G_attn 0 --D_attn 0 --G_ortho 0.0 \
--G_init ortho --D_init ortho --G_ch 32 --D_ch 32 --test_every 10000 \
--save_every 8000 --num_best_copies 0 --num_save_copies 0 --seed 99 \
--sample_every 8000 --G_eval_mode --id imagenet100_unet_noatt_cutmix_cons --gpus 0,1,2,3,4,5,6,7 \
--ema --use_ema --accumulate_stats --num_standing_accumulations 100 \
--unet_mixup --slow_mixup --full_batch_mixup --consistency_loss_and_augmentation --warmup_epochs 200 \
--base_root ./results --data_folder ./dataset"

# ################## TEST for all metric in benchmark ######################
# for DT in "LSUN" "SVHN" "CIFAR100" "Places365" "Texture" "Tiny_imagenet"; do
#     for DET in "8 9 89"; do
#         python -u mcoodcat_on_scood.py ${PARAMS} \
#             --ood_dataset ${DT} --test_detector ${DET} --ind_dataset cifar10 \
#             | tee -a results/cifar10_89.log
#     done
# done

# for DT in "SVHN" "CIFAR10" "LSUN" "Places365" "Texture" "Tiny_imagenet"; do
#     for DET in "8 9 89"; do
#         python -u mcoodcat_on_scood.py ${PARAMS} \
#             --ood_dataset ${DT} --test_detector ${DET} --ind_dataset cifar100 \
#             | tee -a results/cifar100_89.log
#     done
# done


#################### TEST for different mask style ######################
for MASK in "random" "high_fix" "low_fix" "board" "shuffle" "zero_mask" "board0.3"; do
    for DET in "5 8"; do
        for DT in "CIFAR100"; do
            python -u mcoodcat_ablation.py ${PARAMS} --ood_dataset ${DT} \
                --test_detector ${DET} --ind_dataset cifar10 \
                --mask_method ${MASK} | tee -a results/cifar10_100_masking.log
        done
    done
done

#################### TEST for Overhead ############################
python -u mcoodcat_overhead.py ${PARAMS} --ood_dataset "CIFAR10" \
    --test_detector 5 8 --ind_dataset cifar100 | tee -a results/overhead.log