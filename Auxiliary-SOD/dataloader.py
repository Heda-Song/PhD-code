import tensorflow as tf
import numpy as np
import glob
import os
import cv2
from PIL import Image
import pickle as pkl
from tqdm import tqdm
import random

FLAGS = tf.app.flags.FLAGS

class Data_Loader:
  """ Read data from the raw images"""
  def __init__(self):
    self.data_dir = './data/'
    self.img_h = 84 
    self.img_w = 84
    self.img_channels = 4
    self.num_classes = 1

  def load_data_pretrain(self):
    if FLAGS.dataset == 'miniimagenet':
      # load data from pkl file 
      data_path = self.data_dir + FLAGS.dataset + '/' + 'miniimagenet_sal_train.pkl'
      with open(data_path,'rb') as f:
        images_classes = pkl.load(f)
      self.num_classes = len(images_classes)
      images_all, labels = [], []
      for i in range(self.num_classes):
        images_all += list(images_classes[i])
        labels += [i] * len(images_classes[i])
    elif FLAGS.dataset == 'tieredimagenet':
      # load data from pkl file 
      data_path = self.data_dir + FLAGS.dataset + '/'  + 'tieredimagenet_sal_train.pkl'
      with open(data_path,'rb') as f:
        images_classes = pkl.load(f)
      self.num_classes = len(images_classes)
      images_all, labels = [], []
      for i in range(self.num_classes):
        images_all += images_classes[i]
        labels += [i] * len(images_classes[i])
    elif FLAGS.dataset == 'cub':
      # load data from pkl file 
      data_path = self.data_dir + FLAGS.dataset + '/'  + 'CUB_img_sal_train.pkl'
      with open(data_path, 'rb') as f:
        images_classes = pkl.load(f)
      self.num_classes = len(images_classes)
      images_all, labels = [], []
      for i in range(self.num_classes):
        images_all += images_classes[i]
        labels += [i] * len(images_classes[i])

    self.data = list(zip(images_all, labels))

  def load_data_fewshot(self, mode='test'):
    if FLAGS.dataset == 'miniimagenet':
      # load data from pkl file 
      data_path = self.data_dir + FLAGS.dataset + '/' + 'miniimagenet_sal_' + mode + '.pkl'
      with open(data_path,'rb') as f:
        images_all = pkl.load(f)

    elif FLAGS.dataset == 'tieredimagenet':
      # load data from pkl file 
      data_path = self.data_dir + FLAGS.dataset + '/' + 'tieredimagenet_sal_' + mode + '.pkl'
      with open(data_path,'rb') as f:
        images_all = pkl.load(f)

    elif FLAGS.dataset == 'cub':
      # load data from pkl file 
      data_path = self.data_dir + FLAGS.dataset + '/' + 'CUB_img_sal_' + mode + '.pkl'
      with open(data_path, 'rb') as f:
        images_all = pkl.load(f)

    if mode == 'train':
      self.data_fewshot_train = images_all
    elif mode == 'val':
      self.data_fewshot_val = images_all
    elif mode == 'test':
      self.data_fewshot_test = images_all


  def get_next_few_shot_tasks(self, mode):
    if mode == 'train':
      data = self.data_fewshot_train
    elif mode == 'val':
      data = self.data_fewshot_val
    elif mode == 'test':
      data = self.data_fewshot_test

    images_meta_batch = []
    labels_meta_batch = []
    for i in range(FLAGS.meta_batch_size):
      selected_indexes = np.random.permutation(len(data))[:FLAGS.n_way]
      images = []
      for j in range(FLAGS.n_way):
        images_class = data[selected_indexes[j]] # [k,84,84,3]
        selected_img_indexes = np.random.permutation(len(images_class))[:(FLAGS.k_shot+FLAGS.num_query)]
        selected_img = [images_class[k] for k in selected_img_indexes]
        images.append(selected_img)   # [5,16,84,84,3]#

      images = np.reshape(np.transpose(np.stack(images), (1,0,2,3,4)), [-1,self.img_h, self.img_w, self.img_channels])  # [80,84,84,3]
      labels = np.tile(np.arange(FLAGS.n_way).reshape(1,FLAGS.n_way), (FLAGS.num_query, 1)).flatten()

      images_meta_batch.append(images)
      labels_meta_batch.append(labels)

    return images_meta_batch, labels_meta_batch

  def get_next_batch_pretrain(self):

    batch_data = random.sample(self.data, FLAGS.pretrain_batch_size)
    images, labels = zip(*batch_data)

    return images, labels
    