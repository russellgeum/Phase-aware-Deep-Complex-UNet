import numpy as np
import tensorflow as tf


def complex_flatten (real, imag):
    
    real = tf.keras.layers.Flatten()(real)
    imag = tf.keras.layers.Flatten()(imag)
    
    return real, imag


def CReLU (real, imag):
    
    real = tf.keras.layers.ReLU()(real)
    imag = tf.keras.layers.ReLU()(imag)
    
    return real, imag


def zReLU (real, imag):

    real = tf.keras.layers.ReLU()(real)
    imag = tf.keras.layers.ReLU()(imag)
    
    # 각 parts를 값이 있으면 True == 1로 만들고, 값이 0이면 False == 0을 반환
    real_flag = tf.cast(tf.cast(real, tf.bool), tf.float32)
    imag_flag = tf.cast(tf.cast(imag, tf.bool), tf.float32)
    
    # 각 part가 True == 1이면 1 반환, 하나라도 False == 0이면 0반환
    # 그래서 real, imag 중 하나라도 축 위에 값이 있으면 flag는 (0, ...) 이다.
    flag = real_flag * imag_flag

    # flag과 행렬끼리 원소곱을 하여, flag (1, ...)에서는 ReLU를 유지
    # (0, ...) flag에서는 값을 기각한다.
    real = tf.math.multiply(real, flag)
    imag = tf.math.multiply(imag, flag)

    return real, imag


def modReLU (real, imag):
    
    norm = tf.abs(tf.complex(real, imag))
    bias = tf.Variable(np.zeros([norm.get_shape()[-1]]), trainable = True, dtype=tf.float32)
    relu = tf.nn.relu(norm + bias)
    
    real = tf.math.multiply(relu / norm + (1e+5), real)
    imag = tf.math.multiply(relu / norm + (1e+5), imag)
    
    return real, imag


def CLeaky_ReLU (real, imag):

    real = tf.nn.leaky_relu(real)
    imag = tf.nn.leaky_relu(imag)

    return real, imag


def complex_tanh (real, imag):

    real = tf.nn.tanh(real)
    imag = tf.nn.tanh(imag)

    return real, imag


def complex_softmax (real, imag):
    
    magnitude = tf.abs(tf.complex(real, imag))
    magnitude = tf.keras.layers.Softmax()(magnitude)
    
    return magnitude