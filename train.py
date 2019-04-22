"""
The implementation of Cartoon Transfrom using MultiModal ( Cross Domain Transform ).

File author: TJ Park
Date: 13. Feb. 2018
"""

import os, time
import libs.configs.config
from time import gmtime, strftime
from matplotlib import pyplot as plt
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import datasets.datapipe as datapipe
import libs.network.BeautyGAN as model
import libs.utils as utils
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICE'] = ""
    with tf.device('/CPU:0'):
        global_step = tf.train.create_global_step()
        lr = tf.placeholder(tf.float32, name='learning_rate')
        dataA, dataB = datapipe.get_dataset()

    # """ build network """
    net = model.BeautyGAN()
    net.train(dataA, dataB)

    """ setting traning vars and Optimizer """
    global_vars = tf.global_variables()
    train_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(train_vars, print_info=True)

    G_vars = [var for var in train_vars if 'Generator' in var.name]
    D_vars = [var for var in train_vars if 'discriminator' in var.name]
    vgg_vars = [var for var in global_vars if'vgg_19' in var.name]

    G_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(net.Generator_loss,
                                                                                      var_list=G_vars)
    D_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.999).minimize(net.Discriminator_loss,
                                                                                      global_step=global_step,
                                                                                      var_list=D_vars)

    """ summary """
    all_G_loss = tf.summary.scalar("Generator_loss", net.Generator_loss)
    all_D_loss = tf.summary.scalar("Discriminator_loss", net.Discriminator_loss)
    G_A_loss = tf.summary.scalar("G_A_loss", net.Generator_A_loss)
    G_B_loss = tf.summary.scalar("G_B_loss", net.Generator_B_loss)
    D_A_loss = tf.summary.scalar("D_A_loss", net.Discriminator_A_loss)
    D_B_loss = tf.summary.scalar("D_B_loss", net.Discriminator_B_loss)

    summary_op = tf.summary.merge([G_A_loss, G_B_loss, all_G_loss,
                                   D_A_loss, D_B_loss, all_D_loss,
                                   tf.summary.image(name='image/real_A', tensor=net.imageA, max_outputs=1),
                                   tf.summary.image(name='image/real_B', tensor=net.imageB, max_outputs=1),
                                   tf.summary.image(name='image/fake_A', tensor=(net.fake_A+1.)/2, max_outputs=1),
                                   tf.summary.image(name='image/fake_B', tensor=(net.fake_B+1.)/2, max_outputs=1)])

    logdir = os.path.join(FLAGS.summaries_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=tf.Session().graph)

    """ set saver for saving final model and backbone model for restore """
    saver = tf.train.Saver(max_to_keep=3, var_list=train_vars)

    """ Set Gpu Env """
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
        sess.run(init_op)
        vgg_saver = tf.train.Saver(var_list=vgg_vars)
        vgg_saver.restore(sess, FLAGS.VGG19_model_path)
        ckpt = tf.train.get_checkpoint_state(FLAGS.last_checkpoint_model)
        """ resotre checkpoint of Backbone network """
        if ckpt is not None:
            lastest_ckpt = tf.train.latest_checkpoint(FLAGS.last_checkpoint_model)
            print('lastest', lastest_ckpt)
            re_saver = tf.train.Saver(var_list=tf.trainable_variables())
            re_saver.restore(sess, lastest_ckpt)

        """ Generate threads """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            cnt = 0
            face_idx = 1
            eye = [2, 4]
            mouth = [7, 8, 9]
            while not coord.should_stop():
                s_time = time.time()
                current_step = sess.run(global_step)
                learning_rate = 0.0001 * pow(0.1, current_step / 100000)
                feed_dict = {lr : learning_rate}

                """ Update Discriminator """
                _, d_loss = sess.run([D_opt, net.Discriminator_loss], feed_dict=feed_dict)

                """ Update Generator """
                _, g_loss, ori_A, ori_B, gen_A, gen_B = sess.run([G_opt, net.Generator_loss,
                                                                  net.imageA, net.imageB,
                                                                  net.fake_A, net.fake_B], feed_dict=feed_dict)

                duration_time = time.time() - s_time
                print ("""iter %d: time:%.3f(sec), d-loss %.4f, g-loss %.4f """ % (current_step, duration_time, d_loss, g_loss))

                if current_step % 1000 == 0:
                    # write summary
                    summary = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, current_step)
                    summary_writer.flush()

                if current_step % 5000 == 0:
                    # Save a checkpoint
                    save_path = 'output/training/MakeUp_GAN.ckpt'
                    saver.save(sess, save_path, global_step=current_step)

                if current_step + 1 == FLAGS.max_iters:
                    print('max iter : %d, current_step : %d' % (FLAGS.max_iters, current_step))
                    break

        except tf.errors.OutOfRangeError:
            print('Error occured')
        finally:
            saver.save(sess, './output/models/MakeUp_GAN_final.ckpt', write_meta_graph=False)
            coord.request_stop()

        coord.join(threads)
        sess.close()




