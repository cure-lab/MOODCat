## Pretrained models
The pretrained models of MOODCat (cifar10 and cifar100) can be obtained from the following link:
https://drive.google.com/drive/folders/1gQjsiCS7DZ8H8KwsTL_I-jTQes2Ha7b9?usp=sharing

## Data Preparation
Our benchmark settings follow the SCOOD (ICCV'21), please prepare all their
dataset. We assume the data are placed in
`${PROJ_ROOT}/../dataset/scood_benchmark/`. Please check
`ICCV21_SCOOD/scood/data/utils.py` file for the default value.

## Environment
Please check `requirements.txt` file to prepare python packages.

## Command to Run
Please check `run_moodcat.sh` for different experiments we did in our paper.
