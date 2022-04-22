# Leveraging Auxiliary Information for Few-Shot Image Classification

This repo contains the code for Chapter 5, Leveraging Auxiliary Information for Few-Shot Image Classification.

## Dependencies
This code requires the following:
* python python 3.\*
* TensorFlow v1.0+

## Data
Please download the dataset below and put them into the 'data' folder. Or, you could change the path to datasets in dataloader.py. Note that we use pretrained [EGNet](https://github.com/JXingZhao/EGNet) to extract the saliency maps for the images in the following datasets. To make the pretrained EGNet work well with low-resolution images, we pretraine EGNet with 84*84 images.
* miniImageNet: The train, val, and test splits follow [OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING](https://github.com/twitter-research/meta-learning-lstm). Please download the dataset [here](https://drive.google.com/drive/folders/1594hQunYPKySg7KVMonv0KZSAih4dkps?usp=sharing) and put it  under 'data/miniimagenet/' folder. 
* tieredImageNet: The dataset and the train, val and test splits can be found at [Meta-Learning for Semi-Supervised Few-Shot Classification](https://github.com/renmengye/few-shot-ssl-public). To get the saliency maps, please use the pretrained EGNet to extract them.
* CUB: The train, val and test splits follow [A Closer Look at Few-Shot Classification](https://github.com/wyharveychen/CloserLookFewShot). Please download the dataset [here](https://drive.google.com/drive/folders/14OXKJwh_BH7ey5rqOUxiat-ezeHV4xGY?usp=sharing) and put it  under 'data/cub/' folder. 


## Usage
## Episode-based training using 4Conv backbone
### 5-way 1-shot fews-shot training on miniImageNet:
python train.py --mode=few_shot_train --log_dir=logs --dataset=miniimagenet --backbone=4Conv --episodes=300000 --meta_batch_size=3 --k_shot=1 --n_way=5

### 5-way 1-shot fews-shot testing on miniImageNet:
python train.py --mode=few_shot_test --log_dir=logs --dataset=miniimagenet --backbone=4Conv --meta_batch_size=1 --k_shot=1 --n_way=5 --test_model_iter=280000

## Large-scale pretraining using Res12 backbone
### pretraining on miniImageNet:
python train.py --mode=pretrain --log_dir=logs --dataset=miniimagenet --backbone=Res12 --episodes=30000 --pretrain_batch_size=128 --learning_rate=0.1

### few-shot testing based on a pretrained model:
python train.py --mode=few_shot_test_pretrain --log_dir=logs --dataset=miniimagenet --backbone=Res12 --meta_batch_size=1 --k_shot=1 --n_way=5 --test_model_iter=25000

### 5-way 1-shot few-shot training on miniImageNet based on a pretrained model:
python train.py --mode=few_shot_train --log_dir=logs --dataset=miniimagenet --backbone=Res12 --episodes=20000 --meta_batch_size=3 --k_shot=1 --n_way=5 --load_from_pretrain=True --load_model_iter=25000 --learning_rate=0.00001

### few-shot testing based on a few-shot trained model(started from a pretrained model):
python train.py --mode=few_shot_test --log_dir=logs --dataset=miniimagenet --backbone=Res12 --meta_batch_size=1 --k_shot=1 --n_way=5 --load_from_pretrain=True --test_model_iter=8000

### You could change the dataset and other arguements to perform training and testing based on other experimental settings.