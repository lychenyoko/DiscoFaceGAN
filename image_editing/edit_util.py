from copy import deepcopy
import numpy as np
import dnnlib.tflib as tflib


def Get_Manipulated_Wplus_Latent(coef_s, coef_t, input_coef_pl, w_latent_from_input_coef, 
                                 latent_code,  zero_equal_coef = False):
    '''
    Usage:
        A wrapped function to get manipulated W+ code based on two 3DMM coefficients
    
    Args:
        coef_s:                   (np.array) of source coefficient with dimension [1, 254]
        coef_t:                   (np.array) of target coefficient with dimension [1, 254]
        w_latent_from_input_coef: (tf.Operation) the TF operation to get the w plus latent
        input_coef_pl:            (tf.Placeholder) for the coefficient
        latent_code:              (np.array) of the retrieved latent code
        zero_equal_coef:          (np.array) whether to set the identical coefficeints between coef_s and coef_t to zero
    '''
    
    # coefficient dimension: [identity, expression, pose, gamma] = [160, 64, 3, 27]
    
    if zero_equal_coef:
    
        exp_from_coef_s = coef_s[:, 160: 160+64]
        pose_from_coef_s = coef_s[:, 160+64: 160+64+3]
        gamma_from_coef_s = coef_s[:, 160+64+3: 160+64+3+27]
    
    else:
        new_coef_s = deepcopy(coef_s)
        new_coef_t = deepcopy(coef_t)
    
    np_seed = 100
    np.random.seed(np_seed)
    noise_lambda = np.random.normal(size=[1,32])
    wplus_coef_s = tflib.run(w_latent_from_input_coef, 
                              {input_coef_pl: np.concatenate([new_coef_s, noise_lambda], axis = 1)})
    
    wplus_coef_t = tflib.run(w_latent_from_input_coef, 
                              {input_coef_pl: np.concatenate([new_coef_t, noise_lambda], axis = 1)})
    
    edit_latent_code = latent_code + (wplus_coef_t - wplus_coef_s)
    return edit_latent_code
