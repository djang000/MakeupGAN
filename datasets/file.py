import os, glob, shutil
import numpy as np
import random
import scipy.misc as sm
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

if __name__ == "__main__":
    c = []
    for name, color in colors.items():
        rgb_color = mcolors.to_rgb(color)
        c.append(rgb_color)

    image_base_dir = 'makeup_dataset/images'
    seg_base_dir = 'makeup_dataset/segs'
    files = os.listdir(os.path.join(seg_base_dir, 'trainA'))

    """ Get the eye shadow region from Segmentation info """
    for i in files:
        ori_path = os.path.join(seg_base_dir, 'trainA/%s'%i)
        save_path = os.path.join(seg_base_dir, 'test/%s'%i)
        img = sm.imread(ori_path)
        alpha = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=np.float32)
        inds = np.where((img == 2) | (img == 4))
        print(len(inds[0]), len(inds[1]))

        eye_min_x = np.maximum(np.min(inds[0]) - 5, 0)
        eye_max_x = np.minimum(np.max(inds[0]) + 5, 360)
        eye_min_y = np.maximum(np.min(inds[1]) - 5, 0)
        eye_max_y = np.minimum(np.max(inds[1]) + 5, 360)
        alpha[eye_min_x:eye_max_x, eye_min_y:eye_max_y] = c[2]

        inds = np.where((img == 3) | (img == 5))
        print(len(inds[0]), len(inds[1]))
        eye_min_x = np.maximum(np.min(inds[0]) - 5, 0)
        eye_max_x = np.minimum(np.max(inds[0]) + 5, 360)
        eye_min_y = np.maximum(np.min(inds[1]) - 5, 0)
        eye_max_y = np.minimum(np.max(inds[1]) + 5, 360)
        alpha[eye_min_x:eye_max_x, eye_min_y:eye_max_y] = c[4]

        sm.imsave('tt.jpg', alpha)

    # alpha = np.zeros(shape=(361, 361, 3), dtype=np.float32)
    #
    # dataA = os.listdir(os.path.join(image_base_dir, 'trainA'))
    # dataB = os.listdir(os.path.join(image_base_dir, 'trainA'))
    #
    # print(len(dataA), len(dataB), dataA[0])
    #
    # id = np.arange(len(dataB))
    # print(len(id))
    #
    # Aid = id[:len(dataA)]
    # print(len(Aid))
    # b_samples = random.sample(id, 100)
    # a_samples = random.sample(Aid, 100)
    # print(a_samples[:10], b_samples[:10])
    #
    # for s_id in b_samples:
    #     fn = dataB[s_id]
    #     image_path = os.path.join(image_base_dir, 'trainB', fn)
    #     seg_path = os.path.join(seg_base_dir, 'trainB', fn)
    #
    #
    #     if os.path.exists(seg_path):
    #         dst_image_path = os.path.join(image_base_dir, 'testB', fn)
    #         dst_seg_path = os.path.join(seg_base_dir, 'testB', fn)
    #         shutil.move(image_path, dst_image_path)
    #         shutil.move(seg_path, dst_seg_path)
    #     else:
    #         print(image_path, seg_path)
    #         print('FALSE : %s' %fn)
