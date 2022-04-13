import tensorflow as tf
import numpy as np
import dataloader
from dropblock import dropblock
from initializer import ScaledVarianceRandomNormal

FLAGS = tf.app.flags.FLAGS

class Model:
  """ build ResNet-12 model. """
  def __init__(self):
    self.img_size = [84,84,4]
    if FLAGS.mode == 'pretrain':
      self.images = tf.placeholder(tf.float32, [FLAGS.pretrain_batch_size]+self.img_size)
      self.labels = tf.placeholder(tf.int32, [None])
    else:
      self.images = tf.placeholder(tf.float32, [FLAGS.meta_batch_size, FLAGS.n_way*(FLAGS.k_shot+FLAGS.num_query)]+self.img_size)
      self.labels = tf.placeholder(tf.int32, [None, FLAGS.n_way*FLAGS.num_query])
    self.if_train = tf.placeholder(tf.bool, name='if_train')
    self.lr = tf.placeholder(tf.float32)


  def preprocess(self, inp, pretrain=False):
    # augmentation
    delta = 0.4
    paddings = [[0,0],[8,8],[8,8],[0,0]]
    if pretrain:
      images = tf.pad(inp, paddings, 'CONSTANT')
      images = tf.image.random_crop(images, [FLAGS.pretrain_batch_size]+self.img_size)
      # images = tf.image.random_brightness(images, max_delta=delta)
      # images = tf.image.random_contrast(images, lower=1.0-delta, upper=1.0+delta)
      # images = tf.image.random_saturation(images, lower=1.0-delta, upper=1.0+delta)
      images_aug = tf.image.random_flip_left_right(images)
    else:
      images_aug = inp

    # normalisation 
    mean_dic = {'miniimagenet':[120.0112,114.617,102.7689,62.6228], 'tieredimagenet':[120.0112,114.617,102.7689,62.6228], 'cub':[103.432,103.432,99.4967,32.9419]}
    std_dic = {'miniimagenet':[72.161,70.0569,73.7158,106.3546], 'tieredimagenet':[72.161,70.0569,73.7158,106.3546], 'cub':[58.3702,58.5375,62.7546,75.7171]}
    images_norm = (images_aug - mean_dic[FLAGS.dataset]) / std_dic[FLAGS.dataset]
    return images_norm

  def feature_extraction(self, inp, num_classes=None):
    # preprocessing
    if_pretrain = FLAGS.mode == 'pretrain'
    preprocessed_inp = self.preprocess(inp, pretrain=if_pretrain)

    # feedforward the feature extractor
    if FLAGS.backbone == '4Conv':
      embeddings = self.conv_backbone(preprocessed_inp, num_classes=num_classes)
    elif FLAGS.backbone == 'Res12':
      embeddings = self.resnet_backbone(preprocessed_inp, num_classes=num_classes)

    return embeddings

  def few_shot_loss(self):
    """Compute loss based on few-shot learning tasks"""
    loss_all = []
    acc_all = []
    for i in range(FLAGS.meta_batch_size):
      images_i = self.images[i]
      labels_i = self.labels[i]
      output = self.feature_extraction(inp=images_i)
      embeddings = tf.contrib.layers.flatten(output)

      embedding_train = tf.reshape(embeddings[:FLAGS.k_shot*FLAGS.n_way], [FLAGS.k_shot, FLAGS.n_way, -1])
      embedding_train = tf.reduce_mean(embedding_train, axis=0)   # [5,512]
      embedding_query = embeddings[FLAGS.k_shot*FLAGS.n_way:]   # [75,512]

      # Similarity comparison
      scale_factor = 64.0
      if FLAGS.backbone == '4Conv': 
        predictions = self.euclidean_metric(embedding_train, embedding_query) / scale_factor
      elif FLAGS.backbone == 'Res12':
        predictions = self.cosine_metric(embedding_train, embedding_query)

      labels_onehot = tf.one_hot(labels_i, FLAGS.n_way)
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels_onehot)
      loss = tf.reduce_mean(loss)
      accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(predictions), 1), tf.argmax(labels_onehot, 1))
      loss_all.append(loss)
      acc_all.append(accuracies)

    loss_sum = tf.reduce_mean(loss_all)
    accuracies = tf.reduce_mean(acc_all)

    return loss_sum, accuracies

  def pretrain_loss(self, num_classes):
    outputs = self.feature_extraction(inp=self.images, num_classes=num_classes)
    labels_onehot = tf.one_hot(self.labels, num_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels_onehot)
    loss = tf.reduce_mean(loss)
    accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), 1), tf.argmax(labels_onehot, 1))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_sum = loss + FLAGS.weight_decay * tf.reduce_sum(reg_losses)

    return loss_sum, accuracies

  def train(self, loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      if FLAGS.backbone == '4Conv':
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
      elif FLAGS.backbone == 'Res12':
        train_op = tf.train.MomentumOptimizer(self.lr, momentum=0.9, use_nesterov=True).minimize(loss)
    return train_op

  def basic_conv_block(self, inp, num_filter_inp, num_filter, name):
    """ build baisc convolution block for 4Conv backbone"""
    with tf.variable_scope(name) as scope:
      conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
      no_stride = [1,1,1,1]
      conv_stride = [1,2,2,1]
      kernel_size = 3
      kernel = tf.get_variable('conv', [kernel_size, kernel_size, num_filter_inp, num_filter], initializer=conv_initializer, dtype=tf.float32)
      x = tf.nn.conv2d(inp, kernel, no_stride, 'SAME')
      x = tf.contrib.layers.batch_norm(x, scale=True, is_training=self.if_train)
      x = tf.nn.relu(x)
      # Adaptive pooling module
      pooling_stride = [1,2,2,1]
      # output = self.adaptive_pooling(x, 2, num_filter, 'pooling_net'+name[-1])
      # some baselines
      output = tf.nn.max_pool(x, pooling_stride, pooling_stride, padding='SAME')
    return output

  def conv_backbone(self, inp, num_classes=1):
    """Build 4Conv backbone"""
    num_filters = [64,64,64,64]
    with tf.variable_scope('backbone', reuse=tf.AUTO_REUSE):
      block1 = self.basic_conv_block(inp, self.img_size[-1], num_filters[0], 'block1')
      block1 = tf.cond(self.if_train, lambda: dropblock(block1, 0.85, 3), lambda:block1)

      block2 = self.basic_conv_block(block1, num_filters[0], num_filters[1], 'block2')
      block2 = tf.cond(self.if_train, lambda: dropblock(block2, 0.85, 3), lambda:block2)

      block3 = self.basic_conv_block(block2, num_filters[1], num_filters[2], 'block3')
      block3 = tf.cond(self.if_train, lambda: dropblock(block3, 0.85, 3), lambda:block3)

      block4 = self.basic_conv_block(block3, num_filters[2], num_filters[3], 'block4')  # [1,6,6,256]
      block4 = tf.cond(self.if_train, lambda: dropblock(block4, 0.85, 3), lambda:block4)

      if FLAGS.mode == 'pretrain':
        dim = block4.shape.as_list()
        emb_dim = dim[1]*dim[2]*dim[3]
        block4 = self.fc_layer(block4, emb_dim, num_classes)
      return block4

  def basic_resnet_block(self, inp, num_filter_inp, num_filter, name):
    """ build convolution block """
    with tf.variable_scope(name) as scope:
      conv_initializer = ScaledVarianceRandomNormal(factor=0.1)
      l2_reg = tf.contrib.layers.l2_regularizer(1.0)
      no_stride = [1,1,1,1]
      kernel_size = 1
      kernel_shortcut = tf.get_variable('conv_shortcut', [kernel_size, kernel_size, num_filter_inp, num_filter], initializer=conv_initializer, regularizer=l2_reg, dtype=tf.float32)
      shortcut = tf.nn.conv2d(inp, kernel_shortcut, no_stride, 'SAME')
      shortcut = tf.contrib.layers.batch_norm(shortcut, scale=True, is_training=self.if_train)

      x = inp
      kernel_size = 3
      kernel0 = tf.get_variable('conv0', [kernel_size, kernel_size, num_filter_inp, num_filter], initializer=conv_initializer, regularizer=l2_reg, dtype=tf.float32)
      kernel1 = tf.get_variable('conv1', [kernel_size, kernel_size, num_filter, num_filter], initializer=conv_initializer, regularizer=l2_reg, dtype=tf.float32)
      kernel2 = tf.get_variable('conv2', [kernel_size, kernel_size, num_filter, num_filter], initializer=conv_initializer, regularizer=l2_reg, dtype=tf.float32)

      x = tf.nn.conv2d(x, kernel0, no_stride, 'SAME')
      x = tf.contrib.layers.batch_norm(x, scale=True, is_training=self.if_train)
      x = tf.nn.leaky_relu(x, alpha=0.1)

      x = tf.nn.conv2d(x, kernel1, no_stride, 'SAME')
      x = tf.contrib.layers.batch_norm(x, scale=True, is_training=self.if_train)
      x = tf.nn.leaky_relu(x, alpha=0.1)

      x = tf.nn.conv2d(x, kernel2, no_stride, 'SAME')
      x = tf.contrib.layers.batch_norm(x, scale=True, is_training=self.if_train)

      x = x + shortcut
      x = tf.nn.leaky_relu(x, alpha=0.1)

      pooling_stride = [1,2,2,1]
      # output = self.adaptive_pooling(x, 2, num_filter, 'pooling_net'+name[-1])
      output = tf.nn.max_pool(x, pooling_stride, pooling_stride, padding='SAME')
    return output

  def fc_layer(self, inp, inp_dim, num_classes):
    with tf.variable_scope('fc_layer') as scope:
      inp = tf.contrib.layers.flatten(inp)
      fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=tf.float32)
      l2_reg = tf.contrib.layers.l2_regularizer(1.0)
      weights = tf.get_variable('weights', [inp_dim, num_classes], initializer=fc_initializer, regularizer=l2_reg, dtype=tf.float32)
      biases = tf.get_variable('biases', [num_classes], initializer=tf.zeros_initializer(), regularizer=l2_reg, dtype=tf.float32)
      output = tf.add(tf.matmul(inp, weights), biases)
      return output

  def resnet_backbone(self, inp, num_classes=0):
    """ build Res12 backbone """
    with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
      num_filters = [64,128,256,512]
      l2_reg = tf.contrib.layers.l2_regularizer(1.0)

      block1 = self.basic_resnet_block(inp, self.img_size[-1], num_filters[0], 'block1')
      block1 = tf.cond(self.if_train, lambda: dropblock(block1, 0.85, 3), lambda:block1)

      block2 = self.basic_resnet_block(block1, num_filters[0], num_filters[1], 'block2')
      block2 = tf.cond(self.if_train, lambda: dropblock(block2, 0.85, 3), lambda:block2)

      block3 = self.basic_resnet_block(block2, num_filters[1], num_filters[2], 'block3')
      block3 = tf.cond(self.if_train, lambda: dropblock(block3, 0.85, 3), lambda:block3)

      block4 = self.basic_resnet_block(block3, num_filters[2], num_filters[3], 'block4')  # [1,6,6,256]
      block4 = tf.cond(self.if_train, lambda: dropblock(block4, 0.85, 3), lambda:block4)

      embeddings = block4

      # global average pooling
      stride = [1, embeddings.shape.as_list()[-2], embeddings.shape.as_list()[-2], 1]
      output = tf.nn.avg_pool(embeddings, ksize=stride, strides=stride, padding='SAME') # [batch,1,1,2048]
      # fully connected layer for classification
      if FLAGS.mode == 'pretrain':
        output = self.fc_layer(output, num_filters[-1], num_classes)
    
    return output

  def adaptive_pooling(self, inp, stride, num_filter, name):
    """ adaptive pooling operation. """
    with tf.variable_scope(name) as scope:
      shape = inp.shape.as_list()
      conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
      no_stride = [1,1,1,1]
      kernel_size = 3
      kernel = tf.get_variable('conv', [3, 3, num_filter, 1], initializer=conv_initializer, dtype=tf.float32)

      x = tf.nn.conv2d(inp, kernel, no_stride, 'SAME')  # [80,21,21,1]
      x = tf.contrib.layers.batch_norm(x, scale=True, is_training=self.if_train)
      x = tf.nn.sigmoid(x) # [80,21,21,1]

      weighted_maps = inp * x # [80,21,21,64]

      pooling_stride = [1,2,2,1]
      ones_kernel = tf.ones([2,2,shape[-1],1])    
      output = tf.nn.depthwise_conv2d(weighted_maps, ones_kernel, pooling_stride, padding='SAME')
      output = output / (pooling_stride[1]*pooling_stride[2])

      return output
  



  
  def euclidean_metric(self, x, y):
    # x = [5,512]; y = [75,512]
    shape0 = x.shape.as_list()[0]
    shape1 = y.shape.as_list()[0]
    x = tf.tile(tf.expand_dims(x,0), [shape1,1,1]) # [75,5,512]
    y = tf.tile(tf.expand_dims(y,1), [1,shape0,1]) # [75,5,512]
    output = tf.negative(tf.reduce_sum((x - y) ** 2, axis=-1))

    return output

  def cosine_metric(self, x, y):
    # x = [5,512]; y = [75,512]
    shape0 = x.shape.as_list()[0]
    shape1 = y.shape.as_list()[0]
    x = tf.tile(tf.expand_dims(x,0), [shape1,1,1]) # [75,5,512]
    y = tf.tile(tf.expand_dims(y,1), [1,shape0,1]) # [75,5,512]

    normalize_x = tf.nn.l2_normalize(x, axis=-1) # [75,5,512]       
    normalize_y = tf.nn.l2_normalize(y, axis=-1) # [75,5,512]
    output = tf.reduce_sum(tf.multiply(normalize_x,normalize_y), axis=-1)  # [75,5]

    return output
