# Spatial attention-based adaptive pooling for few-shot image classification

This repo contains the code for Chapter 4, Spatial attention-based adaptive pooling for few-shot image classification.

## Dependencies
This code requires the following:
* python python 3.\*
* TensorFlow v1.0+

## Data
Please download the dataset below and put them into the 'data' folder. Or, you could change the path to datasets in dataloader.py
* miniImageNet: We got miniImageNet dataset along with the train, val, and test splits from [OPTIMIZATION AS A MODEL FOR FEW-SHOT LEARNING](https://github.com/twitter-research/meta-learning-lstm). Please put the classes of images for train, val and test under 'data/cub/train/', 'data/cub/val/' and 'data/cub/test/' respectively. To process the raw dataset into train, val and test set, please refer to [MAML](https://github.com/cbfinn/maml).
* tieredImageNet: The dataset and the train, val and test splits can be found at [Meta-Learning for Semi-Supervised Few-Shot Classification](https://github.com/renmengye/few-shot-ssl-public).
* CUB: Download the [CUB dataset](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/browse/index.html). The train, val and test splits follows [A Closer Look at Few-Shot Classification](https://github.com/wyharveychen/CloserLookFewShot). Please put the classes of images for train, val and test under 'data/cub/train/', 'data/cub/val/' and 'data/cub/test/' respectively.


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