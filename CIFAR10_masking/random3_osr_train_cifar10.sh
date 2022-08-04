python random3_osr_train_cifar10.py --dataset CIFAR10 --parallel --batch_size 384  --which_best FID \
--num_D_steps 1  --G_lr 5e-5  --D_lr 2e-4  \
--D_B2 0.999 --G_B2 0.999 --G_attn 0 --D_attn 0 --G_ortho 0.0 \
--G_init ortho --D_init ortho --G_ch 32 --D_ch 32 --test_every 10000 \
--save_every 2000 --num_best_copies 0 --num_save_copies 0 --seed 99 \
--sample_every 2000 --G_eval_mode --id imagenet100_unet_noatt_cutmix_cons --gpus "0,1,2,3,4,5,6,7" \
--ema --use_ema --accumulate_stats --num_standing_accumulations 100 \
--unet_mixup --slow_mixup --full_batch_mixup --consistency_loss_and_augmentation --warmup_epochs 200 \
--base_root ./CIFAR10_osrtask_masked_generator_results_random3 \
--data_folder /data/yangyijun/datasets
