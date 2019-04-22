"""
The implementation of Cartoon Transfrom using MultiModal ( Cross Domain Transform ).

File author: TJ Park
Date: 19. Feb. 2018
"""

import os, time
import libs.configs.config
from time import gmtime, strftime
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import datasets.datapipe as datapipe
import libs.network.BeautyGAN as model

FLAGS = tf.app.flags.FLAGS

def saveimg(img, path):
    invert_img = (img + 1.) /2
    print(np.min(invert_img), np.max(invert_img))
    sm.imsave(path, invert_img)

def inference():
    with tf.device('/CPU:0'):
        inputA = tf.placeholder(tf.float32, shape=[None, None, 3], name='inputA')
        inputB = tf.placeholder(tf.float32,shape=[None, None, 3], name='inputB')

        imageA = datapipe._preprocess_for_test(inputA)
        imageB = datapipe._preprocess_for_test(inputB)

        """ build network """
        net = model.BeautyGAN()
        net.inference(imageA, imageB)

        """ set saver for saving final model and backbone model for restore """
        saver = tf.train.Saver()

        """ Set Gpu Env """
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(FLAGS.last_checkpoint_model)
            """ resotre checkpoint of Backbone network """
            if ckpt is not None:
                lastest_ckpt = tf.train.latest_checkpoint(FLAGS.last_checkpoint_model)
                print('lastest', lastest_ckpt)
                saver.restore(sess, lastest_ckpt)

            base_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name, 'images')
            test_A_files = os.listdir(os.path.join(base_dir, 'testA'))
            test_B_files = os.listdir(os.path.join(base_dir, 'testB'))
            np.random.shuffle(test_A_files)
            np.random.shuffle(test_B_files)
            print(len(test_A_files), len(test_B_files), test_A_files[0], test_B_files[0])
            dataA = sm.imread(base_dir+'/testA/'+test_A_files[0])
            dataB = sm.imread(base_dir+'/testB/'+test_B_files[0])
            print(dataA.shape, dataB.shape)
            feed_dict={inputA : dataA, inputB : dataB}
            recon_A, recon_B, fake_BA, fake_AB = sess.run([net.recon_xa, net.recon_xb,
                                       net.fake_ba, net.fake_ab], feed_dict=feed_dict)

            sm.imsave('oriA.jpg', dataA)
            sm.imsave('oriB.jpg', dataB)
            saveimg(recon_A[0], 'recon_A.jpg')
            saveimg(recon_B[0], 'recon_B.jpg')
            saveimg(fake_BA[0], 'fake_BA.jpg')
            saveimg(fake_AB[0], 'fake_AB.jpg')

            sess.close()

def rand_style_infer():
    with tf.device('/CPU:0'):
        inputA = tf.placeholder(tf.float32, shape=[None, None, 3], name='inputA')
        rand_style = tf.placeholder(tf.float32,shape=[1, 1, 1, 8], name='inputB')

        imageA = datapipe._preprocess_for_test(inputA)

        """ build network """
        net = model.CGNet()
        net.rand_style_infer(imageA, rand_style)

        """ set saver for saving final model and backbone model for restore """
        saver = tf.train.Saver()

        """ Set Gpu Env """
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(FLAGS.last_checkpoint_model)
            """ resotre checkpoint of Backbone network """
            if ckpt is not None:
                lastest_ckpt = tf.train.latest_checkpoint(FLAGS.last_checkpoint_model)
                print('lastest', lastest_ckpt)
                saver.restore(sess, lastest_ckpt)

            base_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)
            test_A_files = os.listdir(os.path.join(base_dir, 'testA'))
            print(len(test_A_files), test_A_files[0])
            dataA = sm.imread(base_dir+'/testA/'+test_A_files[0])
            s_code = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, 1, 8])
            print(dataA.shape)
            feed_dict={inputA : dataA, rand_style : s_code}
            fake_img = sess.run(net.fake_image, feed_dict=feed_dict)

            sm.imsave('oriA.jpg', dataA)
            saveimg(fake_img[0], 'fake_img.jpg')
            sess.close()

if __name__ == '__main__':

    inference()
    # rand_style_infer()




