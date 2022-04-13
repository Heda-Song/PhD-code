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
    self.img_channels = 3
    self.num_classes = 1

  def load_data_pretrain(self):
    if FLAGS.dataset == 'miniimagenet':
      img_dir = self.data_dir + FLAGS.dataset + '/train' 
      classes = sorted(glob.glob(os.path.join(img_dir, '*')))
      self.num_classes = len(classes)
      images_all, labels = [], []
      for i, cls in enumerate(classes): # 80
        img_files = glob.glob(os.path.join(cls, '*.jpg'))
        for j, img_file in enumerate(img_files): # 600
          img = np.array(Image.open(img_file), dtype=np.uint8)
          images_all.append(img)
          labels.append(i)
    elif FLAGS.dataset == 'tieredimagenet':
      img_path = self.data_dir + FLAGS.dataset + '/train_images_png.pkl'
      label_path = self.data_dir + FLAGS.dataset + '/train_labels.pkl'
      with open(img_path, 'rb') as f:
        images_raw = pkl.load(f)                # [448695--[n]]
      with open(label_path, 'rb') as f:
        labels_all = pkl.load(f)
        labels = labels_all["label_specific"] # [448695]
      self.num_classes = np.max(labels) + 1
      images_all = []
      for i, item in tqdm(enumerate(images_raw), desc='decompress'):
        img = cv2.imdecode(item, 1)
        images_all.append(img)
    elif FLAGS.dataset == 'cub':
      img_dir = self.data_dir + FLAGS.dataset + '/train'
      classes = sorted(glob.glob(os.path.join(img_dir, '*')))
      self.num_classes = len(classes)
      images_all, labels = [], []
      for i, cls in enumerate(classes):
        img_files = glob.glob(os.path.join(cls, '*.jpg'))
        for j, img_file in enumerate(img_files): # 600
          img = np.array(Image.open(img_file), dtype=np.uint8) # / 255.0
          images_all.append(img)
          labels.append(i)
    
    self.data = list(zip(images_all, labels))

  def load_data_fewshot(self, mode='test'):
    if FLAGS.dataset == 'miniimagenet':
      img_dir = self.data_dir + FLAGS.dataset + '/' + mode
      classes = sorted(glob.glob(os.path.join(img_dir, '*')))
      images_all = []
      for i, cls in enumerate(classes): # 80
        images_per_class = []
        img_files = glob.glob(os.path.join(cls, '*.jpg'))
        for j, img_file in enumerate(img_files): # 600
          img = np.array(Image.open(img_file), dtype=np.uint8)
          images_per_class.append(img)
        images_all.append(images_per_class)
    elif FLAGS.dataset == 'tieredimagenet':
      img_path = self.data_dir + FLAGS.dataset + '/' + mode + '_images_png.pkl'
      label_path = self.data_dir + FLAGS.dataset + '/' + mode + '_labels.pkl'
      with open(img_path, 'rb') as f:
        images_raw = pkl.load(f)                # [448695--[n]]
      with open(label_path, 'rb') as f:
        labels_all = pkl.load(f)
        labels = labels_all["label_specific"] # [448695]
      
      num_classes = np.max(labels) + 1
      images_all = [[] for _ in range(num_classes)]
      for i, item in tqdm(enumerate(images_raw), desc='decompress'):
        img = cv2.imdecode(item, 1)
        images_all[labels[i]].append(img)

    elif FLAGS.dataset == 'cub':
      img_dir = self.data_dir + FLAGS.dataset + '/' + mode
      classes = sorted(glob.glob(os.path.join(img_dir, '*')))
      images_all = []
      for i, cls in enumerate(classes):
        images_per_class = []
        img_files = glob.glob(os.path.join(cls, '*.jpg'))
        for j, img_file in enumerate(img_files): # 600
          img = np.array(Image.open(img_file), dtype=np.uint8) # / 255.0
          images_per_class.append(img)
        images_all.append(images_per_class)   # [100,k,84,84,3]

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

