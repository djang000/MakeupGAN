import os, math, random
from glob import glob
import numpy as np
from libs.configs import config
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def _flip_image(image):
    return tf.reverse(image, axis=[1])

def _rotate_images(image, gt_mask, seed):
    ratio = seed * 2.0 - 1.0
    angle = (ratio*30.0) * (math.pi / 180.0)
    rotate_image = tf.contrib.image.rotate(image, angle)
    rotate_mask = tf.contrib.image.rotate(gt_mask, angle)
    return rotate_image, rotate_mask


def _preprocess_for_training(dataA, dataB, maskA, maskB):

    resize_imgA = tf.image.resize_images(dataA, [FLAGS.image_size, FLAGS.image_size])
    resize_imgB = tf.image.resize_images(dataB, [FLAGS.image_size, FLAGS.image_size])

    norm_A = tf.cast(resize_imgA, tf.float32) / 127.5 - 1.0
    norm_B = tf.cast(resize_imgB, tf.float32) / 127.5 - 1.0

    maskA.set_shape((361, 361, 1))
    maskB.set_shape((361, 361, 1))

    flip_thresh = tf.random_uniform(shape=[1])
    val = tf.constant(0.5, dtype=tf.float32)
    norm_A, norm_B, mask_A, mask_B = tf.cond(tf.greater_equal(flip_thresh[0], val),
                              lambda : (_flip_image(norm_A),
                                        _flip_image(norm_B),
                                        _flip_image(maskA),
                                        _flip_image(maskB)),
                              lambda : (norm_A, norm_B, maskA, maskB))

    return norm_A, norm_B, mask_A, mask_B

def _preprocess_for_test(image):

    resize_img = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])

    norm_img = tf.cast(resize_img, tf.float32) /127.5 - 1.0

    norm_img = tf.expand_dims(norm_img, 0)
    return norm_img

def get_dataset():
    image_base_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name, 'images')
    mask_base_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name, 'segs')

    trainA_dataset = sorted(glob(os.path.join(image_base_dir, 'trainA/*')))
    trainA_masks = sorted(glob(os.path.join(mask_base_dir, 'trainA/*')))
    trainB_dataset = sorted(glob(os.path.join(image_base_dir, 'trainB/*')))
    trainB_masks = sorted(glob(os.path.join(mask_base_dir, 'trainB/*')))

    num_dataset = max(len(trainA_dataset), len(trainB_dataset))
    print(num_dataset)

    domainA = tf.convert_to_tensor(trainA_dataset)
    domainB = tf.convert_to_tensor(trainB_dataset)
    domainA_mask = tf.convert_to_tensor(trainA_masks)
    domainB_mask = tf.convert_to_tensor(trainB_masks)

    inputA_queue = tf.train.slice_input_producer([domainA, domainA_mask], shuffle=True, name='inputA_queue')
    inputB_queue = tf.train.slice_input_producer([domainB, domainB_mask], shuffle=True, name='inputB_queue')

    imageA_fn = tf.read_file(inputA_queue[0], name='read_imageA')
    maskA_fn = tf.read_file(inputA_queue[1], name='read_maskA')
    imageB_fn = tf.read_file(inputB_queue[0], name='read_imageB')
    maskB_fn = tf.read_file(inputB_queue[1], name='read_maskB')

    dataA = tf.image.decode_png(imageA_fn, channels=3, name='decode_imageA')
    maskA = tf.image.decode_png(maskA_fn, name='decode_maskA')
    dataB = tf.image.decode_png(imageB_fn, channels=3, name='decode_imageB')
    maskB = tf.image.decode_png(maskB_fn, name='decode_maskB')

    aug_imageA, aug_imageB, aug_maskA, aug_maskB = _preprocess_for_training(dataA, dataB, maskA, maskB)


    batch_domainA = tf.train.batch([aug_imageA, aug_maskA],
                              batch_size=FLAGS.batch_size,
                              num_threads=4,
                              capacity=512)

    batch_domainB = tf.train.batch([aug_imageB, aug_maskB],
                              batch_size=FLAGS.batch_size,
                              num_threads=4,
                              capacity=512)

    return batch_domainA, batch_domainB


