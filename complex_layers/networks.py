import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import constraints
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec
from . import initializers as cinitializers



'COMPLEX DENSE'
class complex_Dense(tf.keras.layers.Layer):
    """
    tf.keras.layers.Dense(
        units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
        **kwargs
    )
    """
    def __init__ (self, units = 512,
                        activation = None,
                        use_bias   = True, 
                        kernel_initializer = 'glorot_uniform',
                        bias_initializer   = 'zeros', 
                        kernel_regularizer = None, 
                        bias_regularizer   = None,
                        activity_regularizer = None, 
                        kernel_constraint    = None, 
                        bias_constraint      = None):
        
        super(complex_Dense, self).__init__()

        self.units = units
        self.activation = activation
        self.use_bias   = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer   = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint    = kernel_constraint
        self.bias_constraint      = bias_constraint
        
        
    def build (self, inputs_shape):
        
        self.real_Dense = tf.keras.layers.Dense(units = self.units, 
                                                activation = self.activation, 
                                                use_bias   = self.use_bias, 
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer   = self.bias_initializer, 
                                                kernel_regularizer = self.kernel_regularizer,
                                                bias_regularizer   = self.bias_regularizer,
                                                activity_regularizer = self.activity_regularizer, 
                                                kernel_constraint    = self.kernel_constraint,
                                                bias_constraint      = self.bias_constraint)
        
        self.imag_Dense = tf.keras.layers.Dense(units = self.units, 
                                                activation = self.activation, 
                                                use_bias   = self.use_bias, 
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer   = self.bias_initializer, 
                                                kernel_regularizer = self.kernel_regularizer,
                                                bias_regularizer   = self.bias_regularizer,
                                                activity_regularizer = self.activity_regularizer, 
                                                kernel_constraint    = self.kernel_constraint,
                                                bias_constraint      = self.bias_constraint)
        
        super(complex_Dense, self).build(inputs_shape)
        

    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_Dense(real_inputs) - self.imag_Dense(imag_inputs)
        imag_outputs = self.imag_Dense(real_inputs) + self.real_Dense(imag_inputs)

        return real_outputs, imag_outputs



'COMPLEX CONVOLUTION 2D'
class complex_Conv2D (tf.keras.layers.Layer):
    

    def __init__(self, 
                filters = 32,
                kernel_size = (3, 3), 
                strides = (2, 2), 
                padding = "same",
                activation = None,
                use_bias   = True,
                kernel_initializer = 'glorot_uniform',
                bias_initializer   = 'zeros'):
        
        super(complex_Conv2D, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.activation  = activation
        self.use_bias    = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        
        
    def build (self, inputs_shape):
        
        self.real_Conv2D = tf.keras.layers.Conv2D(filters = self.filters,
                                                kernel_size = self.kernel_size, 
                                                strides = self.strides,
                                                padding = self.padding,
                                                activation = self.activation,
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer = self.bias_initializer) 

        self.imag_Conv2D = tf.keras.layers.Conv2D(filters = self.filters,
                                                kernel_size = self.kernel_size, 
                                                strides = self.strides,
                                                padding = self.padding,
                                                activation = self.activation,
                                                use_bias = self.use_bias,
                                                kernel_initializer = self.kernel_initializer,
                                                bias_initializer = self.bias_initializer) 
        
        super(complex_Conv2D, self).build(inputs_shape)

        
    def call(self, real_inputs, imag_inputs):
        
        real_outputs = self.real_Conv2D(real_inputs) - self.imag_Conv2D(imag_inputs)
        imag_outputs = self.imag_Conv2D(real_inputs) + self.real_Conv2D(imag_inputs)
        
        return real_outputs, imag_outputs



'COMPLEX CONV 2D TRANSPOSE'
class complex_Conv2DTranspose (tf.keras.layers.Layer):


    def __init__(self,  filters = 32,
                        kernel_size = (3, 3), 
                        strides = (2, 2), 
                        padding = "same",
                        activation = None,
                        use_bias   = True,
                        kernel_initializer = 'glorot_uniform',
                        bias_initializer   = 'zeros'):
        
        super(complex_Conv2DTranspose, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides     = strides
        self.padding     = padding
        self.activation  = activation
        self.use_bias    = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer   = bias_initializer
        
        
    def build (self, inputs_shape):

        self.real_Conv2DTranspose = tf.keras.layers.Conv2DTranspose(filters = self.filters,
                                                        kernel_size = self.kernel_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding, 
                                                        activation = self.activation, 
                                                        use_bias = self.use_bias,
                                                        kernel_initializer = self.kernel_initializer, 
                                                        bias_initializer = self.bias_initializer)

        self.imag_Conv2DTranspose = tf.keras.layers.Conv2DTranspose(filters = self.filters,
                                                        kernel_size = self.kernel_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding, 
                                                        activation = self.activation, 
                                                        use_bias = self.use_bias,
                                                        kernel_initializer = self.kernel_initializer, 
                                                        bias_initializer = self.bias_initializer)
        
        super(complex_Conv2DTranspose, self).build(inputs_shape)

        
    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_Conv2DTranspose(real_inputs) - self.imag_Conv2DTranspose(imag_inputs)
        imag_outputs = self.imag_Conv2DTranspose(real_inputs) + self.real_Conv2DTranspose(imag_inputs)

        return real_outputs, imag_outputs



'COMPLEX POOLING'
class complex_MaxPooling (tf.keras.layers.Layer):

    def __init__(self, pool_size = (2, 2), strides   = (1, 1), padding   = "same"):

        super(complex_MaxPooling, self).__init__()

        self.pool_size = pool_size
        self.strides   = strides
        self.padding   = padding


    def build (self, inputs_shape):
        
        self.real_maxpooling = tf.keras.layers.MaxPool2D(pool_size = self.pool_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding)
        
        self.imag_maxpooling = tf.keras.layers.MaxPool2D(pool_size = self.pool_size, 
                                                        strides = self.strides, 
                                                        padding = self.padding)
        
        super(complex_MaxPooling, self).build(inputs_shape)
        

    def call (self, real_inputs, imag_inputs):

        real_outputs = self.real_maxpooling(real_inputs)
        imag_outputs = self.imag_maxpooling(imag_inputs)

        return real_outputs, imag_outputs



'https://github.com/fchollet/keras/blob/master/keras/layers/normalization.py'
'NAIVE BATCH NORMALIZATION
class complex_NaiveBatchNormalization (tf.keras.layers.Layer):
    '''
    tf.keras.layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        beta_initializer='zeros', gamma_initializer='ones',
        moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
        gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
        fused=None, trainable=True, virtual_batch_size=None, adjustment=None, name=None,
        **kwargs
    )
    '''
    def __init__ (selfaxis = -1, 
                momentum = 0.99, 
                epsilon = 0.001, 
                center = True, 
                scale = True,
                beta_initializer = 'zeros', 
                gamma_initializer = 'ones',
                moving_mean_initializer = 'zeros',
                moving_variance_initializer = 'ones',
                beta_regularizer = None, 
                gamma_regularizer = None, 
                beta_constraint = None,
                gamma_constraint = None,
                renorm = False,
                renorm_clipping = None, 
                renorm_momentum = 0.99,
                fused = None, 
                trainable = True, 
                virtual_batch_size = None,
                adjustment = None,
                **kwargs)

        super(complex_NaiveBatchNormalization, self).__init__()

        self.momoentum = momentum
        self.epsilon   = epsilon
        self.center    = center
        self.scale     = scale 
        self.beta_initializer            = beta_initializer
        self.gamma_initializer           = gamma_initializer
        self.moving_mean_initializer     = moving_mean_initializer
        self.moving_variance_initializer = moving_variance_initializer
        self.beta_regularizer            = beta_regularizer
        self.gamma_regularizer           = gamma_regularizer
        self.beta_constraint             = beta_constraint
        self.gamma_constraint            = gamma_constraint
        self.renorm                      = renorm
        self.renorm_clipping             = renorm_clipping
        self.renorm_momentum             = renorm_momentum
        self.fused                       = fused
        self.trainable                   = trainable
        self.virtual_batch_size          = virtual_batch_size
        self.adjustment                  = adjustment

        self.real_batchnormalization = tf.keras.layers.BatchNormalization(momentum = self.momentum,
                                                                        epsilon = self.epsilon,
                                                                        center = self.center,
                                                                        scale = self.scale,
                                                                        beta_initializer = self.beta_initializer,
                                                                        gamma_initializer = self.gamma_initializer,
                                                                        moving_mean_initializer = self.moving_mean_initializer,
                                                                        moving_variance_initializer = self.moving_variance_initializer,
                                                                        beta_regularizer = self.beta_regularizer,
                                                                        gamma_regularizer = self.gamma_regularizer,
                                                                        beta_constraint = self.beta_constraint,
                                                                        gamma_constraint = self.gamma_constraint,
                                                                        renorm = self.renorm,
                                                                        renorm_clipping = self.renorm_clipping,
                                                                        renorm_momentum = self.renorm_momentum,
                                                                        fused = self.fused,
                                                                        trainable = self.trainable,
                                                                        virtual_batch_size = self.virtual_batch_size,
                                                                        adjustment = self.adjustment)

        self.imag_batchnormalization = tf.keras.layers.BatchNormalization(momentum = self.momentum,
                                                                        epsilon = self.epsilon,
                                                                        center = self.center,
                                                                        scale = self.scale,
                                                                        beta_initializer = self.beta_initializer,
                                                                        gamma_initializer = self.gamma_initializer,
                                                                        moving_mean_initializer = self.moving_mean_initializer,
                                                                        moving_variance_initializer = self.moving_variance_initializer,
                                                                        beta_regularizer = self.beta_regularizer,
                                                                        gamma_regularizer = self.gamma_regularizer,
                                                                        beta_constraint = self.beta_constraint,
                                                                        gamma_constraint = self.gamma_constraint,
                                                                        renorm = self.renorm,
                                                                        renorm_clipping = self.renorm_clipping,
                                                                        renorm_momentum = self.renorm_momentum,
                                                                        fused = self.fused,
                                                                        trainable = self.trainable,
                                                                        virtual_batch_size = self.virtual_batch_size,
                                                                        adjustment = self.adjustment)
        

    def call (self, real_inputs, imag_inputs, training = True):

        real_outputs = self.real_batchnormalization (real_inputs, training = training)
        imag_outputs = self.imag_batchnormalization (imag_inputs, training = training)

        return real_outputs, imag_outputs



'COMPLEX BATCH NORMALIZATION MODULE'
def sqrt(shape, dtype = None):
    value = (1 / K.sqrt(2)) * K.ones(shape)
    return value

def initGet(init):
    if init in ['sqrt']:
        return sqrt
    else:
        return initializers.get(init)

def initSet(init):
    if init in [sqrt]:
        return 'sqrt'
    else:
        return initializers.serialize(init)

class ComplexBatchNormalization(tf.keras.layers.Layer):

    def __init__(self,
                 axis = -1,
                 momentum = 0.9,
                 epsilon = 1e-4,
                 center = True,
                 scale = True,
                 beta_initializer = 'zeros',
                 gamma_diag_initializer = 'sqrt',
                 gamma_off_initializer = 'zeros',
                 moving_mean_initializer = 'zeros',
                 moving_variance_initializer = 'sqrt',
                 moving_covariance_initializer = 'zeros',
                 beta_regularizer = None,
                 gamma_diag_regularizer = None,
                 gamma_off_regularizer = None,
                 beta_constraint = None,
                 gamma_diag_constraint = None,
                 gamma_off_constraint = None,
                 **kwargs):

        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer               = cinitializers.get(beta_initializer)
        self.gamma_diag_initializer         = cinitializers.get(gamma_diag_initializer)
        self.gamma_off_initializer          = cinitializers.get(gamma_off_initializer)
        self.moving_mean_initializer        = cinitializers.get(moving_mean_initializer)
        self.moving_variance_initializer    = cinitializers.get(moving_variance_initializer)
        self.moving_covariance_initializer  = cinitializers.get(moving_covariance_initializer)
        self.beta_regularizer               = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer         = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer          = regularizers.get(gamma_off_regularizer)
        self.beta_constraint                = constraints.get(beta_constraint)
        self.gamma_diag_constraint          = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint           = constraints.get(gamma_off_constraint)

    def build(self, input_shape):
        input_shapes = input_shape
        assert(input_shapes[0] == input_shapes[1])
        input_shape = input_shapes[0]
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim = len(input_shape),axes = {self.axis: dim})
        shape = (dim,)
        if self.scale:
            self.gamma_rr = self.add_weight(shape = shape,
                                            name = 'gamma_rr',
                                            initializer = self.gamma_diag_initializer,
                                            regularizer = self.gamma_diag_regularizer,
                                            constraint = self.gamma_diag_constraint)
            self.gamma_rr = self.add_weight(shape = shape,
                                            name = 'gamma_ii',
                                            initializer = self.gamma_diag_initializer,
                                            regularizer = self.gamma_diag_regularizer,
                                            constraint = self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape = shape,
                                            name = 'gamma_ri',
                                            initializer = self.gamma_off_initializer,
                                            regularizer = self.gamma_off_regularizer,
                                            constraint = self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape = shape,
                                              initializer = self.moving_variance_initializer,
                                              name = 'moving_Vrr',
                                              trainable = False)
            self.moving_Vii = self.add_weight(shape = shape,
                                              initializer = self.moving_variance_initializer,
                                              name = 'moving_Vii',
                                              trainable = False)
            self.moving_Vri = self.add_weight(shape = shape,
                                              initializer = self.moving_covariance_initializer,
                                              name = 'moving_Vri',
                                              trainable = False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta_real = self.add_weight(shape = shape,
                                        name = 'beta_real',
                                        initializer = self.beta_initializer,
                                        regularizer = self.beta_regularizer,
                                        constraint = self.beta_constraint)
            self.beta_image = self.add_weight(shape = shape,
                                             name = 'beta_image',
                                             initializer = self.beta_initializer,
                                             regularizer = self.beta_regularizer,
                                             constraint = self.beta_constraint)
            self.moving_mean_real = self.add_weight(shape = shape,
                                               initializer = self.moving_mean_initializer,
                                               name = 'moving_mean_real',
                                               trainable = False)
            self.moving_mean_image = self.add_weight(shape = shape,
                                                    initializer = self.moving_mean_initializer,
                                                    name = 'moving_mean_image',
                                                    trainable = False)
        else:
            self.beta_real = None
            self.beta_image = None
            self.moving_mean_real = None
            self.moving_mean_image = None

        self.built = True

    def call(self, real, imag, training = None):

        assert isinstance(inputs, list)
        input_real, input_image = real, imag
        input_shape = K.int_shape(input_real)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        mu_real  = K.mean(input_real, axis = reduction_axes)
        mu_image = K.mean(input_image, axis = reduction_axes)

        # center_x = x - E[x]
        if self.center:
            centered_real  = input_real - mu_real
            centered_image = input_image - mu_image
        else:
            centered_real  = input_real
            centered_image = input_image
        centered_squared_real  = centered_real ** 2
        centered_squared_image = centered_image ** 2
        centered = K.concatenate([centered_real, centered_image])
        centered_squared = K.concatenate([centered_squared_real, centered_squared_image])

        '''
        Vrr = Cov(R(x), R(x)), Cov(X, Y) = E((X - E(X))(Y - E(Y)))
        Vii = Cov(I(x), I(x))
        Vri = Cov(R(x), I(x))
        '''
        if self.scale:
            Vrr = K.mean(centered_squared_real, axis = reduction_axes, ) + self.epsilon
            Vii = K.mean(centered_squared_image, axis = reduction_axes, ) + self.epsilon
            Vri = K.mean(centered_real * centered_image, axis = reduction_axes, )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')
        input_bn = self.complexBN(centered_real, centered_image, Vrr, Vii, Vri)
        if training in {0, False}: 
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(self.moving_mean_real, mu_real, self.momentum))
                update_list.append(K.moving_average_update(self.moving_mean_image, mu_image, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))
            self.add_update(update_list, inputs)

            def normalize_inference():
                if self.center:
                    inference_centered_real  = input_real - self.moving_mean_real
                    inference_centered_image = input_image - self.moving_mean_image
                else:
                    inference_centered_real  = input_real
                    inference_centered_image = input_image
                return self.complexBN(inference_centered_real, inference_centered_image, Vrr, Vii,Vri)

        return K.in_train_phase(input_bn, normalize_inference, training = training)


    def complexBN(self, centered_real, centered_image, Vrr, Vii, Vri):
        output_real  = centered_real
        output_image = centered_image
        if self.scale:
            t_real,t_image = self.complex_std(centered_real, centered_image, Vrr, Vii, Vri)
            output_real    = self.gamma_rr * t_real + self.gamma_ri * t_image
            output_image   = self.gamma_ri * t_real + self.gamma_ii * t_image
        if self.center:
            output_real  = output_real + self.beta_real
            output_image = output_image + self.beta_image

        return output_real, output_image


    def complex_std(self, centered_real, centered_image, Vrr, Vii, Vri):
        'sqrt of a 2x2 matrix, https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix'
        tau   = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = np.sqrt(delta)
        t = np.sqrt(tau + 2 * s)

        'matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html'
        inverse_st = 1.0 / (s * t)
        Wrr = (Vii +s) * inverse_st
        Wii = (Vrr +s) * inverse_st
        Wri = -Vri * inverse_st

        output_real  = Wrr * centered_real + Wri * centered_image
        output_image = Wri * centered_real + Wii * centered_image

        return output_real, output_image


    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer':                 cinitializers.serialize(self.beta_initializer),
            'gamma_diag_initializer':           cinitializers.serialize(self.gamma_diag_initializer),
            'gamma_off_initializer':            cinitializers.serialize(self.gamma_off_initializer),
            'moving_mean_initializer':          cinitializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':      cinitializers.serialize(self.moving_variance_initializer),
            'moving_covariance_initializer':    cinitializers.serialize(self.moving_covariance_initializer),
            'beta_regularizer':                   regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer':            regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer':             regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint':                    constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint':              constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint':               constraints.serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    