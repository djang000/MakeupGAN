"""
The implementation of Cartoon Transfrom network ).

File author: TJ Park
Date: 14. Feb. 2019
"""

import libs.configs.config
import libs.network.vgg16 as vgg
import tensorflow as tf
import tensorflow.contrib.slim as slim
import libs.utils as utils

FLAGS = tf.app.flags.FLAGS

class BeautyGAN(object):
    def __init__(self):
        """
        Cartoon GAN Construction
        """
        self.weight_decay = FLAGS.weight_decay

    # def inference(self, imgA, imgB ):


    def train(self, dataA, dataB):
        """ hyperParameters """
        self.pw = 0.005
        self.cyc_w = 10.0
        self.gan_w = 1.0
        self.lips_w = 1.0
        self.shadow_w = 1.0
        self.face_w = 0.1

        """ Generator params """
        self.n_res = 4
        self.mlp_dim = 256
        self.n_downsample = 2
        self.n_upsample = 2
        self.style_dim = 8

        """ Discriminator params """
        self.n_scale = 3

        print("##### Train Information #####")
        print("# dataset : ", FLAGS.dataset_name)
        print("# batch_size : ", FLAGS.batch_size)
        print("# max_iters : ", FLAGS.max_iters)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)
        print("# Style dimension : ", self.style_dim)
        print("# MLP dimension : ", self.mlp_dim)
        print("# Down sample : ", self.n_downsample)
        print("# Up sample : ", self.n_upsample)

        print()

        print("##### Discriminator #####")
        print("# Multi-scale Dis : ", self.n_scale)

        # build Network
        self.imageA = dataA[0]
        self.maskA = dataA[1]
        self.imageB = dataB[0]
        self.maskB = dataB[1]

        self._build_graph()
        self.losses()

    def inference(self, xa, xb):
        self.fake_ba, self.fake_ab = self.generator(xa, xb)
        self.recon_xa, self.recon_xb = self.generator(self.fake_ba, self.fake_ab, reuse=True)

    def _build_graph(self):
        """ Encoder graph """
        self.fake_A, self.fake_B = self.generator(self.imageA, self.imageB)
        self.recon_A, self.recon_B = self.generator(self.fake_A, self.fake_B, reuse=True)

        """ Discriminator """
        self.real_A_logit = self.Discriminator(self.imageA, scope='discriminator_A')
        self.fake_A_logit = self.Discriminator(self.fake_A, reuse=True, scope='discriminator_A')

        self.real_B_logit = self.Discriminator(self.imageB, scope='discriminator_B')
        self.fake_B_logit = self.Discriminator(self.fake_B, reuse=True, scope='discriminator_B')

    def generator(self, imgA, imgB, reuse=False):
        with slim.arg_scope(training_scope(weight_decay=self.weight_decay)):
            with tf.variable_scope('Generator', reuse=reuse):
                with tf.variable_scope('Encoder', reuse=reuse):
                    encoder_A = self.Encoder_A(imgA, reuse=reuse, scope='encoder_A')
                    encoder_B = self.Encoder_B(imgB, reuse=reuse, scope='encoder_B')

                    concat_x = tf.concat([encoder_A, encoder_B], axis=3, name='concat_AB')
                    x = slim.conv2d(concat_x, 256, [3, 3], stride=2, normalizer_fn=slim.instance_norm, scope='conv_0')

                """ residual block """
                with tf.variable_scope('residual_block', reuse=reuse):
                    x = self.residual_block(x, scope='rb_1')
                    x = self.residual_block(x, scope='rb_2')
                    x = self.residual_block(x, scope='rb_3')
                    x = self.residual_block(x, scope='rb_4')

                with tf.variable_scope('Decoder', reuse=reuse):
                    gen_A = self.Decoder_A(x, reuse=reuse, scope='decoder_A')
                    gen_B = self.Decoder_B(x, reuse=reuse, scope='decoder_B')

                return gen_A, gen_B

    def Encoder_A(self, img, reuse=False, scope='encoder_A'):
        with tf.variable_scope(scope, reuse=reuse):
            x = slim.conv2d(img, 64, [7, 7], stride=1, normalizer_fn=slim.instance_norm, scope='conv_0')
            x = slim.conv2d(x, 128, [3, 3], stride=2, normalizer_fn=slim.instance_norm, scope='conv_1')
            return x

    def Encoder_B(self, img, reuse=False, scope='encoder_B'):
        with tf.variable_scope(scope, reuse=reuse):
            x = slim.conv2d(img, 64, [7, 7], stride=1, normalizer_fn=slim.instance_norm, scope='conv_0')
            x = slim.conv2d(x, 128, [3, 3], stride=2, normalizer_fn=slim.instance_norm, scope='conv_1')
            return x

    def residual_block(self, x, reuse=False, scope='residual_block'):
        with tf.variable_scope(scope, reuse=reuse):
            y = slim.conv2d(x, 256, [3, 3], normalizer_fn=slim.instance_norm, scope='res1')
            y = slim.conv2d(y, 256, [3, 3], normalizer_fn=slim.instance_norm, activation_fn=None, scope='res2')
            return x + y

    def Decoder_A(self, x, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            x = self.up_sampling(x, 128, scope='up_sampling_1')
            x = self.up_sampling(x, 64, scope='up_sampling_2')
            x = slim.conv2d(x, 3, [7, 7], activation_fn=None, scope='generate_logit')
            x = tf.tanh(x)
            return x

    def Decoder_B(self, x, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            x = self.up_sampling(x, 128, scope='up_sampling_1')
            x = self.up_sampling(x, 64, scope='up_sampling_2')
            x = slim.conv2d(x, 3, [7, 7], activation_fn=None, scope='generate_logit')
            x = tf.tanh(x)
            return x

    def up_sampling(self, x, ch, scope='up_sampling'):
        with tf.variable_scope(scope):
            _, h, w, _ = x.get_shape().as_list()
            new_size = [h*2-1, w*2-1]
            up_x = tf.image.resize_nearest_neighbor(x, size=new_size, name='resize_nn')
            x = slim.conv2d(up_x, ch, [5, 5], normalizer_fn=slim.instance_norm, scope='conv')
            return x

    def Discriminator(self, feature, reuse=False, scope='discriminator'):
        D_logit=[]
        with tf.variable_scope(scope, reuse=reuse):
            x_ = feature
            for scale in range(self.n_scale):
                with tf.variable_scope('scale_%d' % scale, reuse=reuse):
                    x = tf.nn.leaky_relu(self.conv2d(x_, 64, [3, 3], stride=1, scope='conv_1_1'))
                    x = self.conv2d(x, 64, [3, 3], stride=2, scope='conv_1_2')
                    x = tf.nn.leaky_relu(slim.batch_norm(x))

                    x = self.conv2d(x, 128, [3, 3], stride=1, scope='conv_2_1')
                    x = tf.nn.leaky_relu(slim.batch_norm(x))
                    x = self.conv2d(x, 128, [3, 3], stride=2, scope='conv_2_2')
                    x = tf.nn.leaky_relu(slim.batch_norm(x))

                    x = self.conv2d(x, 256, [3, 3], stride=1, scope='conv_3_1')
                    x = tf.nn.leaky_relu(slim.batch_norm(x))
                    x = self.conv2d(x, 256, [3, 3], stride=2, scope='conv_3_2')
                    x = tf.nn.leaky_relu(slim.batch_norm(x))

                    x = self.conv2d(x, 512, [3, 3], stride=1, scope='conv_4_1')
                    x = tf.nn.leaky_relu(slim.batch_norm(x))
                    x = self.conv2d(x, 512, [3, 3], stride=2, scope='conv_4_2')
                    x = tf.nn.leaky_relu(slim.batch_norm(x))

                    x = self.conv2d(x, 1, [1, 1], stride=1, padding='VALID', scope='conv_5')

                    D_logit.append(x)
                    x_ = tf.nn.avg_pool(x_, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            return D_logit

    def conv2d(self, x, out_ch, kernel_size, stride, padding='SAME', scope=None):
        in_ch = x.get_shape().as_list()[-1]
        strides = [1, stride, stride, 1]
        with tf.variable_scope(scope):
            w = tf.get_variable("w", [kernel_size[0], kernel_size[0], in_ch, out_ch], initializer=slim.initializers.xavier_initializer())
            biases = tf.get_variable("b", [out_ch], initializer=tf.zeros_initializer())
            norm_w = spectral_norm(w)
            conv = tf.nn.conv2d(x, norm_w, strides=strides, padding=padding)
            out = tf.nn.bias_add(conv, biases)
            return out

    def losses(self):
        """ 1. Reconstruction Loss (L1 Loss) """
        """ Makeup Loss """
        face_loss_A, shadow_loss_A, lips_loss_A, bg_loss_A = self.calc_makeup_loss(self.fake_B, self.imageA, self.maskA, self.imageB, self.maskB)
        face_loss_B, shadow_loss_B, lips_loss_B, bg_loss_B = self.calc_makeup_loss(self.fake_A, self.imageB, self.maskB, self.imageA, self.maskA)
        makeup_loss = self.lips_w * (lips_loss_A + lips_loss_B) + \
                      self.shadow_w * (shadow_loss_A + shadow_loss_B) + \
                      self.face_w * (face_loss_A + face_loss_B)

        """ 2. Cycle Loss (L1 Loss) """
        cycA_loss = tf.reduce_mean(tf.abs(self.recon_A - self.imageA))
        cycB_loss = tf.reduce_mean(tf.abs(self.recon_B - self.imageB))

        """ 3. Adversial Loss (LS GAN Loss) """
        adv_GA = 0
        adv_GB = 0
        adv_DA = 0
        adv_DB = 0
        for i in range(self.n_scale):
            """ Generator loss """
            adv_GA += tf.reduce_mean(tf.squared_difference(self.fake_A_logit[i], 1.0))
            adv_GB += tf.reduce_mean(tf.squared_difference(self.fake_B_logit[i], 1.0))

            """ Discriminator loss """
            ra_loss = tf.reduce_mean(tf.squared_difference(self.real_A_logit[i], 1.0))
            fa_loss = tf.reduce_mean(tf.square(self.fake_A_logit[i]))
            adv_DA += tf.add(ra_loss, fa_loss)

            rb_loss = tf.reduce_mean(tf.squared_difference(self.real_B_logit[i], 1.0))
            fb_loss = tf.reduce_mean(tf.square(self.fake_B_logit[i]))
            adv_DB += tf.add(rb_loss, fb_loss)

        """ 4. Perceptual Loss using VGG19 network """
        _, e = self.vgg_19(self.vgg_preprocessing(self.fake_B, name='output_AB'))
        gen_featureA = e['vgg_19/conv5/conv5_3']

        _, e = self.vgg_19(self.vgg_preprocessing(self.imageA, name='output_A'), reuse=True)
        featureA = e['vgg_19/conv5/conv5_3']

        _, e = self.vgg_19(self.vgg_preprocessing(self.fake_A, name='output_BA'), reuse=True)
        gen_featureB = e['vgg_19/conv5/conv5_3']

        _, e = self.vgg_19(self.vgg_preprocessing(self.imageB, name='output_B'), reuse=True)
        featureB = e['vgg_19/conv5/conv5_3']

        perceptual_A_Loss = tf.reduce_mean(tf.squared_difference(gen_featureA, featureA))
        perceptual_B_Loss = tf.reduce_mean(tf.squared_difference(gen_featureB, featureB))


        """ Each loss of the data A and the data B """
        self.Generator_A_loss = makeup_loss + \
                                self.gan_w * adv_GA +\
                                self.cyc_w * cycA_loss + \
                                self.pw * perceptual_A_Loss


        self.Generator_B_loss = makeup_loss + \
                                self.gan_w * adv_GB + \
                                self.cyc_w * cycB_loss + \
                                self.pw * perceptual_B_Loss

        self.Discriminator_A_loss = self.gan_w * adv_DA
        self.Discriminator_B_loss = self.gan_w * adv_DB

        """ Total Loss """
        self.Generator_loss = self.Generator_A_loss + self.Generator_B_loss
        self.Discriminator_loss = self.Discriminator_A_loss + self.Discriminator_B_loss


    def vgg_preprocessing(self, img, name):
        vgg_mean = [103.939, 116.779, 123.68]   # BGR

        vggInput_AB = (img + 1.0) * 127.5
        r, g, b = tf.split(vggInput_AB, 3, 3)
        output = tf.concat(values=[b-vgg_mean[0], g-vgg_mean[1], r-vgg_mean[2]], axis=3, name=name)

        return output


    def vgg_19(self,
               inputs,
               is_training=False,
               reuse=False,
               scope='vgg_19'):
        """Oxford Net VGG 19-Layers version E Example.
        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          is_training: whether or not the model is being trained.
          scope: Optional scope for the variables.
        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the non-dropped-out input to the logits layer (if num_classes is 0 or
            None).
          end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], trainable = is_training, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable = is_training, scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], trainable = is_training, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], trainable = is_training, scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], trainable = is_training, scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return net, end_points

    def calc_makeup_loss(self, fake_img, src_img, src_mask, ref_img, ref_mask):
        face_mask, face_HM = tf.py_func(utils.crop_face_mask,
                                                  [fake_img[0], src_mask[0], ref_img[0], ref_mask[0]],
                                                  [tf.float32, tf.float32])

        shadow_mask, shadow_HM = tf.py_func(utils.crop_eye_mask,
                                                      [fake_img[0], src_mask[0], ref_img[0], ref_mask[0]],
                                                      [tf.float32, tf.float32])

        lips_mask, lips_HM = tf.py_func(utils.crop_mouth_mask,
                                                  [fake_img[0], src_mask[0], ref_img[0], ref_mask[0]],
                                                  [tf.float32, tf.float32])

        gen_bg_mask, img_bg_mask = tf.py_func(utils.crop_BG_mask,
                                        [fake_img[0], src_img[0], src_mask[0]],
                                        [tf.float32, tf.float32])

        self.fhm = face_HM
        self.lhm = lips_HM
        self.shm = shadow_HM
        self.gen_bg_mask = gen_bg_mask
        self.img_bg_mask = img_bg_mask

        face_loss = tf.reduce_mean(tf.squared_difference(face_mask, face_HM))
        shadow_loss = tf.reduce_mean(tf.squared_difference(shadow_mask, shadow_HM))
        lips_loss = tf.reduce_mean(tf.squared_difference(lips_mask, lips_HM))
        bg_loss = tf.reduce_mean(tf.squared_difference(gen_bg_mask, img_bg_mask))
        return face_loss, shadow_loss, lips_loss, bg_loss

def spectral_norm(w, iter=1):
    w_shape = w.get_shape().as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iter):
        """ power iteration
        usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def training_scope(weight_intitializer=slim.initializers.xavier_initializer(),
                   weight_decay=0.00004):

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=weight_intitializer,
        activation_fn=tf.nn.relu) :
        with slim.arg_scope([slim.conv2d], padding='SAME', weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
            return sc
