import scipy
import scipy.signal
import scipy.io.wavfile
import librosa
from librosa.display import *

import os, ssl
import natsort
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.utils import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import*
from tensorflow.keras.activations import *
from tensorflow.keras.initializers import *
from tensorflow.python.client import device_lib

from complex_layers.STFT import *
from complex_layers.networks import *
from complex_layers.activations import *



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'시퀀스를 이미지로 변환하거나, 이미지를 시퀀스로 변환하는 모듈'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def convert2image (inputs):
    
    outputs = tf.transpose(inputs, perm = (0, 2, 1))
    outputs = tf.reshape(outputs, (-1, 512, 64, 1))
    
    return outputs


def convert2sequnce (inputs):
    
    outputs = tf.transpose(inputs, (0, 2, 1, 3))
    outputs = tf.squeeze(outputs, axis = 3)
    
    return outputs
    

def mask_processing (real, imag):
    
    norm  = tf.sqrt(tf.square(real) + tf.square(imag))
    magnitude  = tf.tanh(norm)

    normalize_phase_real = tf.divide(real, norm)
    normalize_phase_imag = tf.divide(imag, norm)

    mask_real = tf.multiply(magnitude, normalize_phase_real)
    mask_imag = tf.multiply(magnitude, normalize_phase_imag)
    
    return mask_real, mask_imag



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'with Naive complex_BatchNormalization module'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def encoder_module (real, imag, filters, kernel_size, strides, training = True):
    
    conv_real, conv_imag = complex_Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    out_real, out_imag   = CLeaky_ReLU(conv_real, conv_imag)
    out_real, out_imag   = complex_NaiveBatchNormalization()(conv_real, conv_imag, training = training)
    
    return out_real, out_imag, conv_real, conv_imag


def decoder_module (real, imag, concat_real, concat_imag, filters, kernel_size, strides, training = True):

    real = concatenate([real, concat_real], axis = 3)
    imag = concatenate([imag, concat_imag], axis = 3)
    deconv_real, deconv_imag = conplex_Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    deconv_real, deconv_imag = CLeaky_ReLU(deconv_real, deconv_imag)
    deconv_real, deconv_imag = complex_NaiveBatchNormalization()(deconv_real, deconv_imag, training = training)
    
    return deconv_real, deconv_imag


def center_module (real, imag, filters, kernel_size, strides, training = True):

    real, imag = conplex_Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    real, imag = CLeaky_ReLU(real, imag)
    real, imag = complex_NaiveBatchNormalization()(real, imag, training = training)
    
    return real, imag


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 'with Covariance complex_BatchNormalization module'
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# def cov_encoder_module (real, imag, filters, kernel_size, strides, training = True):
    
#     conv_real, conv_imag = complex_Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
#     conv_real, conv_imag = CLeaky_ReLU(conv_real, conv_imag)

#     batchnorm_inputs     = tf.concat([conv_real, conv_imag], axis = 3)
#     batchnorm_outputs    = complex_BatchNormalization()(batchnorm_inputs, training = training)

#     out_real = batchnorm_outputs[:, :, :, 0:1]
#     out_imag = batchnorm_outputs[:, :, :, 1:2]
    
#     return out_real, out_imag, conv_real, conv_imag


# def cov_decoder_module (real, imag, concat_real, concat_imag, filters, kernel_size, strides, training = True):

#     real = concatenate([real, concat_real], axis = 3)
#     imag = concatenate([imag, concat_imag], axis = 3)
#     deconv_real, deconv_imag = conplex_Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
#     deconv_real, deconv_imag = CLeaky_ReLU(deconv_real, deconv_imag)

#     batchnorm_inputs     = tf.concat([deconv_real, deconv_imag], axis = 3)
#     batchnorm_outputs    = complex_BatchNormalization()(batchnorm_inputs, training = training)

#     deconv_real = batchnorm_outputs[:, :, :, 0:1]
#     deconv_imag = batchnorm_outputs[:, :, :, 1:2]
    
#     return deconv_real, deconv_imag


# def cov_center_module (real, imag, filters, kernel_size, strides, training = True):

#     real, imag = conplex_Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
#     real, imag = CLeaky_ReLU(real, imag)

#     batchnorm_inputs     = tf.concat([real, imag], axis = 3)
#     batchnorm_inputs     = complex_BatchNormalization()(batchnorm_inputs, training = training)

#     real = batchnorm_inputs[:, :, :, 0:1]
#     imag = batchnorm_inputs[:, :, :, 1:2]
    
#     return real, imag


'READ FILE PATH'
def walk_filename (file_path = os.path.join("./datasets\_noisy")):

    file_list = []

    for root, dirs, files in tqdm(os.walk(file_path)):
        for fname in files:
            if fname == "desktop.ini" or fname == ".DS_Store": continue 

            full_fname = os.path.join(root, fname)
            file_list.append(full_fname)

    file_list = natsort.natsorted(file_list, reverse = False)

    return file_list


'IMPLEMENT SHORT TIME FOURIER TRANSFORM'
def stft(data, n_fft, hop_length):

    result = []
    
    for index in range (len(data)):
        result.append(librosa.core.stft(y = data[index], 
                                        n_fft = n_fft,
                                        hop_length = hop_length, 
                                        win_length = n_fft))
                      
    return result


'INVERSE STFT'
def istft (data, hop_length):
    
    result = []
    data = np.reshape(data, (len(data), len(data[0]), len(data[0])))
    
    for index in range (len(data)):
        result.append(librosa.core.istft(data[index],
                                        hop_length = hop_length))
        
    return result


'BINARY MASK'
def binary_mask (clean, noisy, alpha = 1.0, criteria = 0.5):
    
    eps = np.finfo(np.float).eps

    mask = np.divide(np.abs(clean)**alpha, (eps + np.abs(noisy))**alpha)
    mask[np.where(mask >= criteria)] = 1.
    mask[np.where(mask <= criteria)] = 0.
    
    return mask


'RATIO MASK'
def ratio_mask (clean, noisy, beta = 0.5):
        
    eps = np.finfo(np.float).eps
    noisy = noisy - clean
    
    clean = np.abs(clean)**2
    noisy = np.abs(noisy)**2
    
    mask = np.divide(clean, (eps + clean + noisy)) ** beta

    return mask


'SHOW MASK'
def show_mask (data, title, fig_size = (7, 7)):

    plt.rcParams["figure.figsize"] = (len(data), len(data[0]))
    plt.rcParams.update({'font.size': 10})
    
    data = np.reshape(data, (len(data), len(data[0])))
    data = np.flip(data, axis = 0)
    
    plt.title(title)
    plt.imshow(data)
    plt.show()


'SHOW SPECTOGRAM'
def show_spectogram (data, title = None, samrpling_rate = 16000, hop_length = 256, shape = (4, 8), colorbar_optional = False):

    if data.ndim == 3:
        data = np.squeeze(data, axis = 0)
    
    plt.rcParams["figure.figsize"] = shape
    plt.rcParams.update({'font.size': 10})
    
    data = np.reshape(data, (len(data), len(data[0])))
    data = librosa.amplitude_to_db(np.abs(data), ref=np.max)
    librosa.display.specshow(data, y_axis = 'hz', x_axis = 'time', sr = 16000, hop_length = 256)

    if title:
        plt.title(title)
    else: pass
    
    if colorbar_optional == True: 
        plt.colorbar(format = '%+2.0f dB')
    elif colorbar_optional == False:
        pass

    plt.show()


'SAVE SPEECH'
def save_file (path, speech, sr = 16000):
    for index, data in enumerate (speech):
        scipy.io.wavfile.write(str(path) + "_" + str(index+1) + ".wav", rate = sr, data = data)