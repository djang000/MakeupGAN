from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

##########################
#                  restore
##########################
tf.app.flags.DEFINE_string(
    'VGG19_model_path', 'pretrained_models/vgg_19.ckpt',
    'Path to pretrained model')

tf.app.flags.DEFINE_string(
    'summaries_dir', './output/summaries/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_model', './output/models/Cartoon_GAN_final.ckpt',
    'Path to checkpoint model')

tf.app.flags.DEFINE_string(
    'last_checkpoint_model', './output/training',
    'Path to latest checkpoint model')

##########################
#                  dataset
##########################
tf.app.flags.DEFINE_string(
    'dataset_name', 'makeup_dataset',
    'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'phase', 'train',
    'The name of the train or test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'datasets',
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1,
    'number of images in a mini-batch')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 7200,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'max_iters', 300000,
    'max iterations')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.0001, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'adam',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.5,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          'The learning rate.')

tf.app.flags.DEFINE_float('gan_w', 1.0,
                          'weight of adversarial loss.')

tf.app.flags.DEFINE_float('recon_x_w', 10.0,
                          'weight of image reconstruction loss.')

tf.app.flags.DEFINE_float('recon_s_w', 1.0,
                          'weight of style reconstruction loss.')

tf.app.flags.DEFINE_float('recon_c_w', 1.0,
                          'weight of content reconstruction loss.')

tf.app.flags.DEFINE_float('cyc_w', 0.0,
                          'weight of cycle consistency loss.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 100.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'image_size', 361,
    'resize image so that the min edge equals to image_size')

tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS