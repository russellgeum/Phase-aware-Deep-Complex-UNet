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
'Converting from sequence to image, Converting from image to seqeunce'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def convert2image (real, imag):
    
    real = tf.transpose(real, perm = (0, 2, 1))
    imag = tf.transpose(imag, perm = (0, 2, 1))

    real = tf.reshape(real, (-1, 512, 64, 1))
    imag = tf.reshape(imag, (-1, 512, 64, 1))
    
    return real, imag


def convert2sequnce (real, imag):
    
    real = tf.transpose(real, (0, 2, 1, 3))
    imag = tf.transpose(imag, (0, 2, 1, 3))

    real = tf.squeeze(real, axis = 3)
    imag = tf.squeeze(imag, axis = 3)
    
    return real, imag
    

def mask_processing (real, imag, stft_real, stft_imag):
    
    magnitude = tf.tanh(tf.sqrt(tf.square(real) + tf.square(imag)))
    unit_real = tf.divide(real, tf.sqrt(tf.square(real) + tf.square(imag)))
    unit_imag = tf.divide(imag, tf.sqrt(tf.square(real) + tf.square(imag)))

    mask_real = tf.multiply(magnitude, unit_real)
    mask_imag = tf.multiply(magnitude, unit_imag)

    enhancement_real = stft_real * mask_real - stft_imag * mask_imag
    enhancement_imag = stft_real * mask_imag + stft_imag * mask_real
    
    return enhancement_real, enhancement_imag



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'with Naive complex_BatchNormalization module'
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def encoder_module (real, imag, filters, kernel_size, strides, training = True):
    
    conv_real, conv_imag = complex_Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    out_real, out_imag   = CLeaky_ReLU(conv_real, conv_imag)
    out_real, out_imag   = complex_NaiveBatchNormalization()(conv_real, conv_imag, training = training)
    
    return out_real, out_imag, conv_real, conv_imag



def decoder_module (real, imag, concat_real, concat_imag, filters, kernel_size, strides, training = True):

    if concat_real == None and concat_imag == None:
        pass
    else:
        real = concatenate([real, concat_real], axis = 3)
        imag = concatenate([imag, concat_imag], axis = 3)
    deconv_real, deconv_imag = complex_Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides)(real, imag)
    deconv_real, deconv_imag = CLeaky_ReLU(deconv_real, deconv_imag)
    deconv_real, deconv_imag = complex_NaiveBatchNormalization()(deconv_real, deconv_imag, training = training)
    
    return deconv_real, deconv_imag


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 'with Naive complex_BatchNormalization module'
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# class encoder (tf.keras.layers.Layer):


#     def __init__ (self, filters, kernel_size, strides, name = "Encoder"): 

#         super(encoder, self).__init__()

#         self.filters     = filters
#         self.kernel_size = kernel_size
#         self.strides     = strides

#         self.complex_Conv2D = complex_Conv2D(filters = self.filters, kernel_size = self.kernel_size, strides = self.strides)
#         self.complex_NaiveBatchNormalization = complex_NaiveBatchNormalization()


#     def call (self, real, imag, training = True):

#         conv_real, conv_imag = self.complex_Conv2D(real, imag)
#         out_real, out_imag   = CLeaky_ReLU(conv_real, conv_real)
#         out_real, out_imag   = self.complex_NaiveBatchNormalization(conv_real, conv_imag, training)

#         return out_real, out_imag, conv_real, conv_imag



# class decoder (tf.keras.layers.Layer):


#     def __init__ (self, filters, kernel_size, strides, name = "Decoder"):

#         super(decoder, self).__init__()

#         self.filters     = filters
#         self.kernel_size = kernel_size
#         self.strides     = strides

#         self.complex_Conv2DTranspose = complex_Conv2DTranspose(filters = self.filters, kernel_size = self.kernel_size, strides = self.strides)
#         self.complex_NaiveBatchNormalization = complex_NaiveBatchNormalization()


#     def call (self, real, imag, concat_real = None, concat_imag = None, training = True):

#         if concat_real is None and concat_imag is None:
#             pass

#         else:
#             real = concatenate([real, concat_real], axis = 3)
#             imag = concatenate([imag, concat_imag], axis = 3)

#         deconv_real, deconv_imag = self.complex_Conv2DTranspose(real, imag)
#         deconv_real, deconv_imag = CLeaky_ReLU(deconv_real, deconv_imag)
#         deconv_real, deconv_imag = self.complex_NaiveBatchNormalization(deconv_real, deconv_imag, training = training)
        
#         return deconv_real, deconv_imag



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