from renderer.face_decoder import Face3D
import dnnlib.tflib as tflib

import os
import time

import tensorflow as tf
from scipy.io import loadmat
from PIL import Image
import numpy as np

def Generate_Render_Face(discofacegan_save_dir):
    '''
    Usage:
        Render the faces based on reconstruction coefficient
    '''

    coeff_dir = os.path.join(discofacegan_save_dir, 'coeff')

    render_img_dir = os.path.join(discofacegan_save_dir, 'render_img')
    if not os.path.exists(render_img_dir):
        os.mkdir(render_img_dir)    
    
    tflib.init_tf()
    with tf.device('/cpu:0'):
        FaceRender = Face3D()
        INPUTcoeff_pl = tf.placeholder(tf.float32, name='input_coeff', shape=[1,254])
        INPUTcoeff_w_t = tf.concat([INPUTcoeff_pl, tf.zeros([1,3])], axis = 1)
        render_img,render_mask,render_landmark,_ = FaceRender.Reconstruction_Block(INPUTcoeff_w_t,256,1,progressive=False)

    aligned_coef_list = os.listdir(coeff_dir)

    start_time = time.time()
    for coef_file in aligned_coef_list:
        coef_file_path = os.path.join(coeff_dir, coef_file)
        coef = loadmat(coef_file_path)['coeff']
        render_img_np = tflib.run(render_img, {INPUTcoeff_pl: coef[:, :254]})

        r_img = Image.fromarray(render_img_np[0].astype(np.uint8))
        render_img_file_path = os.path.join(render_img_dir, 'r_' + coef_file.replace('mat', 'png'))
        r_img.save(render_img_file_path)

    end_time = time.time()
    print('Total Process Time: ' + str(round(end_time - start_time, 2)))

