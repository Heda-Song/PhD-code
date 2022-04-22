import tensorflow as tf
import numpy as np
from dataloader import Data_Loader
from models import Model
#import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

np.set_printoptions(precision=4,suppress=True,threshold=np.inf)

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', 'logs',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('dataset', 'miniimagenet',
                            """miniimagenet, tieredimagenet, cub.""") 
tf.app.flags.DEFINE_integer('episodes', 100000,
                            """Number of epoches to run.""")
tf.app.flags.DEFINE_string('mode', 'few_shot_train',
                            """few_shot_train, few_shot_test for episodic training and testing of few-shot learning tasks; pretrain for large-scale pretraining; few_shot_test_pretrain; """) 
tf.app.flags.DEFINE_string('backbone', '4Conv',
                            """4Conv and Res12.""")
tf.app.flags.DEFINE_integer('pretrain_batch_size', 128,
                            """Batch size for pretraining.""")  
tf.app.flags.DEFINE_integer('meta_batch_size', 3,
                            """meta batch size.""")    
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                            """Initial learning rate for updating.""")
tf.app.flags.DEFINE_float('weight_decay', 0.001,
                            """weight_decay""")     
tf.app.flags.DEFINE_integer('k_shot', 1,
                            """number of training samples per class.""")  
tf.app.flags.DEFINE_integer('n_way', 5,
                            """number of classes for a classification task.""")  
tf.app.flags.DEFINE_integer('num_query', 6,
                            """Number of queries.""")  
tf.app.flags.DEFINE_integer('test_model_iter', -1,
                            """The model from training iteration to load for testing.""")  
tf.app.flags.DEFINE_boolean('load_from_pretrain', False,
                            """load pretrained model""")             
tf.app.flags.DEFINE_integer('load_model_iter', -1,
                            """The model from pretraining to load for few_shot training.""")                                                                                      
def few_shot_train():
  """episodic training for few-shot learning."""
  if FLAGS.load_from_pretrain:
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 100
    VAL_EPISODE = 1000
  else:
    SAVE_INTERVAL = 10000
    PRINT_INTERVAL = 1000
    VAL_EPISODE = 1000

  with tf.Graph().as_default():

    # load data
    print("Load data")
    data_loader = Data_Loader()
    data_loader.load_data_fewshot(mode='train')
    data_loader.load_data_fewshot(mode='val')

    # Build the model
    print("Build the model")
    model = Model()

    # Calculate loss and accuracy
    loss, accuracy = model.few_shot_loss()

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss)

    sess = tf.Session()

    # Summary operation
    tf.summary.scalar('training_loss', loss)
    tf.summary.scalar('training_accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # initialisation
    sess.run(tf.global_variables_initializer())

    # load from pretrained model
    if FLAGS.load_from_pretrain:
      variables = tf.contrib.framework.get_variables_to_restore()
      variables_to_restore = [v for v in variables if 'linear' not in v.name]
      variables_to_restore = [v for v in variables_to_restore if 'Momentum' not in v.name]
      saver = tf.train.Saver(variables_to_restore)
      saver.restore(sess, FLAGS.log_dir + '/pretrain/' + FLAGS.dataset + '/' + FLAGS.backbone + '_pretrain_model_' + str(FLAGS.load_model_iter))

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    if FLAGS.load_from_pretrain:
      save_path = FLAGS.log_dir + '/few_shot_load_pretrain/' + FLAGS.dataset + '/'
    else:
      save_path = FLAGS.log_dir + '/few_shot/' + FLAGS.dataset + '/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    print("Start training:")
    losses, accuracies = 0, 0
    val_best_acc = 0
    lr_step = FLAGS.episodes // 5
    for itr in range(FLAGS.episodes):
      lr = FLAGS.learning_rate * (0.5**(itr//lr_step))

      images_batch, labels_batch = data_loader.get_next_few_shot_tasks(mode='train')
      feed_dict = {model.images: images_batch, model.labels: labels_batch, model.if_train: True, model.lr:lr}
      result = sess.run([train_op, summary_op, loss, accuracy], feed_dict)

      losses += result[-2]
      accuracies += result[-1]

      if (itr!=0) and itr % PRINT_INTERVAL == 0:
        print_str = "episode: " + str(itr) + ", loss: " + str(losses/PRINT_INTERVAL) + ", accuracy: " + str(accuracies/PRINT_INTERVAL)
        print(print_str)
        train_writer.add_summary(result[1], itr)
        losses, accuracies = 0, 0

      if (itr!=0) and itr % SAVE_INTERVAL == 0:
        val_accuracy = []
        for _ in range(VAL_EPISODE):
          images_batch, labels_batch = data_loader.get_next_few_shot_tasks(mode='val')
          feed_dict = {model.images: images_batch, model.labels: labels_batch, model.if_train: False}
          result = sess.run([accuracy, loss], feed_dict)
          val_accuracy.append(result[0])
        val_avg_acc = np.mean(val_accuracy)
        print("validation acc:", val_avg_acc)
        if val_avg_acc > val_best_acc:
          print("saving current model...")
          if FLAGS.load_from_pretrain:
            saver.save(sess, FLAGS.log_dir + '/few_shot_load_pretrain/' + FLAGS.dataset + '/' + FLAGS.backbone + '_' + str(FLAGS.n_way) + '_way' + str(FLAGS.k_shot) + '_shot_' + str(itr), write_meta_graph=False)
          else:
            saver.save(sess, FLAGS.log_dir + '/few_shot/' + FLAGS.dataset + '/' + FLAGS.backbone + '_' + str(FLAGS.n_way) + '_way' + str(FLAGS.k_shot) + '_shot_' + str(itr), write_meta_graph=False)
          val_best_acc = val_avg_acc

    if FLAGS.load_from_pretrain:
      saver.save(sess, save_path + FLAGS.backbone + '_' + str(FLAGS.n_way) + '_way' + str(FLAGS.k_shot) + '_shot_' + str(itr+1))
    else:
      saver.save(sess, save_path + FLAGS.backbone + '_' + str(FLAGS.n_way) + '_way' + str(FLAGS.k_shot) + '_shot_' + str(itr+1))
    print("stop training")


def few_shot_test():
  TEST_EPOCH = 6000

  # load data
  print("Load test data")
  data_loader = Data_Loader()
  data_loader.load_data_fewshot(mode='test')

  # Build the model
  print("Build the model")
  model = Model()

  # Calculate loss and accuracy
  loss, accuracy = model.few_shot_loss()

  sess = tf.Session()
  
  # restore trained model
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
  if FLAGS.mode == 'few_shot_test':
    if FLAGS.load_from_pretrain:
      model_path = FLAGS.log_dir + '/few_shot_load_pretrain/' + FLAGS.dataset + '/' + FLAGS.backbone + '_' + str(FLAGS.n_way) + '_way' + str(FLAGS.k_shot) + '_shot_' + str(FLAGS.test_model_iter)
    else:
      model_path = FLAGS.log_dir + '/few_shot/' + FLAGS.dataset + '/' + FLAGS.backbone + '_' + str(FLAGS.n_way) + '_way' + str(FLAGS.k_shot) + '_shot_' + str(FLAGS.test_model_iter)
  elif FLAGS.mode == 'few_shot_test_pretrain':
    model_path = FLAGS.log_dir + '/pretrain/' + FLAGS.dataset + '/' + FLAGS.backbone + '_pretrain_model_' + str(FLAGS.test_model_iter)
  saver.restore(sess, model_path)
  print('Load model from {}'.format(model_path))

  print("Few-shot Testing:")
  test_accuracy = []
  for itr in range(TEST_EPOCH):
    images_batch, labels_batch = data_loader.get_next_few_shot_tasks(mode='test')
    feed_dict = {model.images: images_batch, model.labels: labels_batch, model.if_train: False}
    result = sess.run([accuracy, loss], feed_dict)
    test_accuracy.append(result[0])
    # print(np.mean(test_accuracy))
  
  mean = np.mean(test_accuracy)
  stds = np.std(test_accuracy)
  ci95 = 1.96*stds/np.sqrt(TEST_EPOCH)
  print("test_accuray:", mean)
  print("95_confidence_interval:", ci95)


def pretrain():
  SAVE_INTERVAL = 1000
  PRINT_INTERVAL = 100
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for miniimagenet(train and val).
    print("Load images")
    data_loader = Data_Loader()
    data_loader.load_data_pretrain()

    # Build a Graph that computes the logits predictions from the inference model.
    print("Build the graph")
    model = Model()

    # Calculate loss and accuracy.
    loss, accuracy = model.pretrain_loss(data_loader.num_classes)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss)

    sess = tf.Session()

    # Summary operation
    tf.summary.scalar('training_loss', loss)
    tf.summary.scalar('training_accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    save_path = FLAGS.log_dir + '/pretrain/' + FLAGS.dataset + '/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("Start training:")
    losses, accuracies = 0, 0
    lr_step = FLAGS.episodes // 3
    for itr in range(FLAGS.episodes):
      lr = FLAGS.learning_rate * (0.1**(itr//lr_step))

      images_batch, labels_batch = data_loader.get_next_batch_pretrain()
      feed_dict = {model.images: images_batch, model.labels: labels_batch, model.if_train: True, model.lr:lr}
      result = sess.run([train_op, summary_op, loss, accuracy], feed_dict)

      losses += result[-2]
      accuracies += result[-1]

      if (itr!=0) and itr % PRINT_INTERVAL == 0:
        print_str = "episode: " + str(itr) + ", loss: " + str(losses/PRINT_INTERVAL) + ", accuracy: " + str(accuracies/PRINT_INTERVAL)
        print(print_str)
        train_writer.add_summary(result[1], itr)
        losses, accuracies = 0, 0

      if (itr!=0) and itr % SAVE_INTERVAL == 0:
        saver.save(sess, save_path + FLAGS.backbone + '_pretrain_model_' + str(itr), write_meta_graph=False)

    saver.save(sess, save_path + FLAGS.backbone + '_pretrain_model_' + str(itr+1))
    print("stop pretraining")


def main(argv=None):
  if FLAGS.mode == 'few_shot_train' or FLAGS.mode == 'few_shot_load_train':
    few_shot_train()
  elif FLAGS.mode == 'pretrain':
    pretrain()
  else:
    few_shot_test()

if __name__ == '__main__':
  tf.app.run()
