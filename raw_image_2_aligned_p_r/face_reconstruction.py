import tensorflow as tf
import cv2
from scipy.io import savemat
from PIL import Image
import numpy as np

import os

from preprocess.preprocess_utils import *
from renderer import face_decoder
from training.networks_recon import R_Net

from pathlib import Path
file_path = Path(__file__).parent
R_net_weights = str((file_path / '''../training/pretrained_weights/recon_net/FaceReconModel''').resolve())
BFM09_FILE = str((file_path / '''../renderer/BFM face model/BFM_model_front_gan.mat''').resolve())

def Extract_Coeff_And_Align_Image(config, image_path, lm_path, save_path):
    '''
    Usage:
        Extract the coefficient and align image for DiscoFaceGAN/ours model processing 
    
    Args:
        image_path: (str) Training image path
        lm_path: (str) Deteced landmark path
        save_path: (str) Saved root path
    '''
    
    # create save path for aligned images and extracted coefficients
    if not os.path.exists(os.path.join(save_path,'img')):
        os.makedirs(os.path.join(save_path,'img'))
    if not os.path.exists(os.path.join(save_path,'coeff')):
        os.makedirs(os.path.join(save_path,'coeff'))

    # Load BFM09 face model
    if not os.path.isfile(BFM09_FILE):
        transferBFM09()

    # Load standard landmarks for alignment
    lm3D = load_lm3d()


    # Build reconstruction model
    with tf.Graph().as_default() as graph:

        images = tf.placeholder(name = 'input_imgs', shape = [None,224,224,3], dtype = tf.float32)
        Face3D = face_decoder.Face3D() # analytic 3D face formation process
        coeff = R_Net(images,is_training=False) # 3D face reconstruction network

        with tf.Session(config = config) as sess:

            var_list = tf.trainable_variables()
            g_list = tf.global_variables()

            # Add batch normalization params into trainable variables 
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list +=bn_moving_vars

            # Create saver to save and restore weights
            resnet_vars = [v for v in var_list if 'resnet_v1_50' in v.name]
            res_fc = [v for v in var_list if 'fc-id' in v.name or 'fc-ex' in v.name or 'fc-tex' in v.name or 'fc-angles' in v.name or 'fc-gamma' in v.name or 'fc-XY' in v.name or 'fc-Z' in v.name or 'fc-f' in v.name]
            resnet_vars += res_fc

            saver = tf.train.Saver(var_list = var_list,max_to_keep = 100)
            saver.restore(sess,R_net_weights)

            for file in os.listdir(os.path.join(image_path)):
                if file.endswith('png'):
                    try:
                        print(file)

                        # load images and landmarks
                        image = Image.open(os.path.join(image_path,file)).convert('RGB')
                        if not os.path.isfile(os.path.join(lm_path,file.replace('png','txt'))):
                            continue
                        lm = np.loadtxt(os.path.join(lm_path,file.replace('png','txt')))
                        lm = np.reshape(lm,[5,2])

                        # align image for 3d face reconstruction
                        align_img,_,_ = Preprocess(image,lm,lm3D) # 512*512*3 RGB image
                        align_img = np.array(align_img)

                        align_img_ = align_img[:,:,::-1] #RGBtoBGR
                        align_img_ = cv2.resize(align_img_,(224,224)) # input image to reconstruction network should be 224*224
                        align_img_ = np.expand_dims(align_img_,0)

                        coef = sess.run(coeff,feed_dict = {images: align_img_})

                        print('Get Coef: ' + str(coef.shape))
                        print(np.linalg.norm(coef))


                        # align image for GAN training
                        # eliminate translation and rescale face size to proper scale
                        rescale_img = crop_n_rescale_face_region(align_img,coef) # 256*256*3 RGB image
                        coef = np.squeeze(coef,0)

                        # save aligned images and extracted coefficients
                        cv2.imwrite(os.path.join(save_path,'img',file),rescale_img[:,:,::-1])
                        savemat(os.path.join(save_path,'coeff',file.replace('.png','.mat')),{'coeff':coef})
                    except:
                        print('Oops, Error')
                    
                    print(' ')
