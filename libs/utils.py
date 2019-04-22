"""
The implementation of Cartoon Transfrom network ).

File author: TJ Park
Date: 18. Feb. 2019
"""
import copy
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as plt
import libs.configs.config
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def saveimg(img, path):
    invert_img = (img + 1.) /2
    sm.imsave(path, invert_img)

def crop_BG_mask(generated_img, imageA, maskA):
    BG_Ainds = np.where(((maskA > 0) & (maskA < 4)) | ((maskA >6) & (maskA <10)))

    gen_mask = copy.copy(generated_img)
    img_mask = copy.copy(imageA)

    gen_mask[BG_Ainds[:2]] = -1
    img_mask[BG_Ainds[:2]] = -1

    return gen_mask, img_mask

def crop_face_mask(imageA, maskA, imageB, maskB):

    faceA = np.full(shape=[FLAGS.image_size,FLAGS.image_size, 3], fill_value=-1.0, dtype=np.float32)
    face_Ainds = np.where(maskA == 1)
    faceA[face_Ainds[:2]] = imageA[face_Ainds[:2]]

    faceB = np.full(shape=[FLAGS.image_size,FLAGS.image_size, 3], fill_value=-1.0, dtype=np.float32)
    face_Binds = np.where(maskB == 1)
    faceB[face_Binds[:2]] = imageB[face_Binds[:2]]

    face_HM = Histogram_Matching(faceA, faceB)

    return faceA, face_HM

def crop_eye_mask(imageA, maskA, imageB, maskB):
    offset = 10
    alphaA = np.full(shape=[FLAGS.image_size,FLAGS.image_size, 3], fill_value=-1.0, dtype=np.float32)
    leye_inds = np.where((maskA == 2) | (maskA == 4))
    reye_inds = np.where((maskA == 3) | (maskA == 5))

    if len(leye_inds[0]) > 0 or len(leye_inds[1]) > 0:
        eye_min_x = np.maximum(np.min(leye_inds[0]) - offset, 0)
        eye_max_x = np.minimum(np.max(leye_inds[0]) + offset, FLAGS.image_size-1)
        eye_min_y = np.maximum(np.min(leye_inds[1]) - offset, 0)
        eye_max_y = np.minimum(np.max(leye_inds[1]) + offset, FLAGS.image_size-1)
        w = np.max(leye_inds[0]) - np.min(leye_inds[0]) + 1
        h = np.max(leye_inds[1]) - np.min(leye_inds[1]) + 1
        if w > 10 and h > 10:
            alphaA[eye_min_x:eye_max_x, eye_min_y:eye_max_y] = imageA[eye_min_x:eye_max_x, eye_min_y:eye_max_y]

    if len(reye_inds[0]) > 0 or len(reye_inds[1]) > 0:
        eye_min_x = np.maximum(np.min(reye_inds[0]) - offset, 0)
        eye_max_x = np.minimum(np.max(reye_inds[0]) + offset, FLAGS.image_size-1)
        eye_min_y = np.maximum(np.min(reye_inds[1]) - offset, 0)
        eye_max_y = np.minimum(np.max(reye_inds[1]) + offset, FLAGS.image_size-1)
        w = np.max(reye_inds[0]) - np.min(reye_inds[0]) + 1
        h = np.max(reye_inds[1]) - np.min(reye_inds[1]) + 1
        if w > 10 and h > 10:
            alphaA[eye_min_x:eye_max_x, eye_min_y:eye_max_y] = imageA[eye_min_x:eye_max_x, eye_min_y:eye_max_y]

    A_eyes_inds = np.where((maskA == 4) | (maskA == 5))
    alphaA[A_eyes_inds[:2]] = -1.0

    alphaB = np.full(shape=[FLAGS.image_size,FLAGS.image_size, 3], fill_value=-1.0, dtype=np.float32)
    leye_inds = np.where((maskB == 2) | (maskB == 4))
    reye_inds = np.where((maskB == 3) | (maskB == 5))

    if len(leye_inds[0]) > 0 or len(leye_inds[1]) > 0:
        eye_min_x = np.maximum(np.min(leye_inds[0]) - offset, 0)
        eye_max_x = np.minimum(np.max(leye_inds[0]) + offset, FLAGS.image_size-1)
        eye_min_y = np.maximum(np.min(leye_inds[1]) - offset, 0)
        eye_max_y = np.minimum(np.max(leye_inds[1]) + offset, FLAGS.image_size-1)
        w = np.max(leye_inds[0]) - np.min(leye_inds[0]) + 1
        h = np.max(leye_inds[1]) - np.min(leye_inds[1]) + 1
        if w > 10 and h > 10:
            alphaB[eye_min_x:eye_max_x, eye_min_y:eye_max_y] = imageB[eye_min_x:eye_max_x, eye_min_y:eye_max_y]

    if len(reye_inds[0]) > 0 or len(reye_inds[1]) > 0:
        eye_min_x = np.maximum(np.min(reye_inds[0]) - offset, 0)
        eye_max_x = np.minimum(np.max(reye_inds[0]) + offset, FLAGS.image_size-1)
        eye_min_y = np.maximum(np.min(reye_inds[1]) - offset, 0)
        eye_max_y = np.minimum(np.max(reye_inds[1]) + offset, FLAGS.image_size-1)
        w = np.max(reye_inds[0]) - np.min(reye_inds[0]) + 1
        h = np.max(reye_inds[1]) - np.min(reye_inds[1]) + 1
        if w > 10 and h > 10:
            alphaB[eye_min_x:eye_max_x, eye_min_y:eye_max_y] = imageB[eye_min_x:eye_max_x, eye_min_y:eye_max_y]

    B_eyes_inds = np.where((maskB == 4) | (maskB == 5))
    alphaB[B_eyes_inds[:2]] = -1.0

    eye_HM = Histogram_Matching(alphaA, alphaB)

    return alphaA, eye_HM

def crop_mouth_mask(imageA, maskA, imageB, maskB):
    alphaA = np.full(shape=[FLAGS.image_size, FLAGS.image_size, 3], fill_value=-1.0, dtype=np.float32)

    mouth_Ainds = np.where((maskA == 7) | (maskA == 9))
    alphaA[mouth_Ainds[:2]] = imageA[mouth_Ainds[:2]]

    alphaB = np.full(shape=[FLAGS.image_size, FLAGS.image_size, 3], fill_value=-1.0, dtype=np.float32)
    mouth_Binds = np.where((maskB == 7) | (maskB == 9))
    alphaB[mouth_Binds[:2]] = imageB[mouth_Binds[:2]]

    mouth_HM = Histogram_Matching(alphaA, alphaB)

    return alphaA, mouth_HM

def Histogram_Matching(src_img, ref_img):
    inds = np.where(src_img==-1)
    out_img = np.full(shape=[FLAGS.image_size, FLAGS.image_size, 3], fill_value=-1.0, dtype=np.float32)

    for d in range(src_img.shape[2]):
        src_gray = src_img[:, :, d]
        dst_gray = ref_img[:, :, d]

        source = src_gray.ravel()
        template = dst_gray.ravel()
        s_vals, bin_idx, s_cnt = np.unique(source, return_inverse=True, return_counts=True)
        t_vals, t_cnt = np.unique(template, return_counts=True)

        src_cdf = np.cumsum(s_cnt).astype(np.float64)
        src_cdf /= src_cdf[-1]

        dst_cdf = np.cumsum(t_cnt).astype(np.float64)
        dst_cdf /= dst_cdf[-1]

        interp_img = np.interp(src_cdf, dst_cdf, t_vals)
        out_img[:, :, d] = interp_img[bin_idx].reshape(out_img.shape[:2])
        out_img[inds] = -1

    return out_img


def HM(src_img, dst_img):
    inds = np.where(src_img==0)
    out_img = np.zeros_like(src_img)

    fig = plt.figure()
    gs = plt.GridSpec(4, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[1, :])
    ax5 = fig.add_subplot(gs[2, :])
    ax6 = fig.add_subplot(gs[3, :])
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    for d in range(src_img.shape[2]):
        src_gray = src_img[:, :, d]
        dst_gray = dst_img[:, :, d]

        source = src_gray.ravel()
        template = dst_gray.ravel()
        s_vals, bin_idx, s_cnt = np.unique(source, return_inverse=True, return_counts=True)
        t_vals, t_cnt = np.unique(template, return_counts=True)

        src_cdf = np.cumsum(s_cnt).astype(np.float64)
        src_cdf /= src_cdf[-1]

        dst_cdf = np.cumsum(t_cnt).astype(np.float64)
        dst_cdf /= dst_cdf[-1]

        interp_img = np.interp(src_cdf, dst_cdf, t_vals)
        out_img[:, :, d] = interp_img[bin_idx].reshape(out_img.shape[:2])
        out = out_img[:, :, d]

        if d==0:
            m_vals, m_cnt = np.unique(out.ravel(), return_counts=True)
            out_cdf = np.cumsum(m_cnt).astype(np.float64)
            out_cdf /= out_cdf[-1]

            ax4.set_title('Red')
            ax4.plot(s_vals, src_cdf*100, '-r', lw=3, label='source')
            ax4.plot(t_vals, dst_cdf*100, '-k', lw=3, label='template')
            ax4.plot(m_vals, out_cdf*100, '--r', lw=2, label='matched')
            ax4.set_xlim(s_vals[0], s_vals[-1])
            ax4.set_xlabel('Pixel Value')
            ax4.set_ylabel('Cumulative %')
            ax4.legend(loc=5)

        elif d==1:
            m_vals, m_cnt = np.unique(out.ravel(), return_counts=True)
            out_cdf = np.cumsum(m_cnt).astype(np.float64)
            out_cdf /= out_cdf[-1]
            ax5.set_title('Green')
            print(s_vals)
            ax5.plot(s_vals, src_cdf*100, '-r', lw=3, label='source')
            ax5.plot(t_vals, dst_cdf*1000, '-k', lw=3, label='template')
            ax5.plot(m_vals, out_cdf*1000, '--r', lw=3, label='matched')
            ax5.set_xlim(s_vals[0], s_vals[-1])
            ax5.set_xlabel('Pixel Value')
            ax5.set_ylabel('Cumulative %')
            ax5.legend(loc=5)
        else:
            m_vals, m_cnt = np.unique(out.ravel(), return_counts=True)
            out_cdf = np.cumsum(m_cnt).astype(np.float64)
            out_cdf /= out_cdf[-1]
            ax6.set_title('Blue')
            ax6.plot(s_vals, src_cdf*1000, '-r', lw=3, label='source')
            ax6.plot(t_vals, dst_cdf*1000, '-k', lw=3, label='template')
            ax6.plot(m_vals, out_cdf*1000, '--r', lw=3, label='matched')
            ax6.set_xlim(s_vals[0], s_vals[-1])
            ax6.set_xlabel('Pixel Value')
            ax6.set_ylabel('Cumulative %')
            ax6.legend(loc=5)

    out_img[inds] = 0

    ax1.imshow(src_img)
    ax1.set_title('Source')

    ax2.imshow(dst_img)
    ax2.set_title('Template')

    ax3.imshow(out_img)
    ax3.set_title('Matched')

    plt.show()

    return out_img
