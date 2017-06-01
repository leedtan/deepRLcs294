# RESUED CODE FROM https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
import math
import numpy as np 
import tensorflow as tf
import keras as keras
from keras.layers.local import LocallyConnected1D as Local
import copy

from tensorflow.python.framework import ops
def concat(dim, objects, name=None):
    if name is None:
        return tf.concat(objects, dim)
    else:
        return tf.concat(objects, dim, name = None)

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def conv_regional(input_, output_dim, 
           k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
           name="conv2d"):
    
    in_len = input_.get_shape()[1].value
    inbound1, inbound2 = int(in_len//4), int(in_len//2)
    
    conv_all = conv2d(input_, output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + 'all')
    
    conv_0 = conv2d(input_[:,:inbound1, :, :], output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '0')
    
    conv_1 = conv2d(input_[:,inbound1:inbound2, :, :], output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '1')
    
    conv_2 = conv2d(input_[:,inbound2:, :, :], output_dim,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '2')
    conv_regions = tf.concat((conv_0, conv_1, conv_2), 1)
    conv = tf.concat((conv_all, conv_regions), 3)
    
    return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def deconv2d_audio(input_, output_shape,
             k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d", padding='SAME'):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w_1 = tf.get_variable('w', [k_h, 1, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv_1 = tf.nn.conv2d_transpose(input_, w_1, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1], padding=padding)

        w_2 = tf.get_variable('w2', [k_h, 1, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv_2 = tf.nn.conv2d_transpose(input_, w_2, output_shape=[int(o) for o in output_shape],
                                strides=[1, d_h, d_w, 1], padding=padding)
        '''
        testing
        deconv_2 = tf.concat([tf.zeros((3,4,1,4)),tf.ones((3,4,1,4))],2)
        s = tf.Session()
        s.run(tf.global_variables_initializer())
        deconv_3 = tf.concat([deconv_2[:,:,1:,:], deconv_2[:,:,:1,:]] , 2)
        s.run(deconv_2)
        s.run(deconv_3)
        '''
        deconv_2 = tf.concat([deconv_2[:,:,1:,:], deconv_2[:,:,:1,:]] , 2)
        deconv = deconv_1 + deconv_2
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv
    
def deconv_regional(input_, output_shape,
             k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d", padding='SAME'):
    in_len = input_.get_shape()[1].value
    inbound1, inbound2 = int(in_len//4), int(in_len//2) 
    o1, o2 = int(output_shape[1]//4), int(output_shape[1]//2)
    
    deconv_all = deconv2d_audio(input_, output_shape,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + 'all', padding=padding)
    outs = output_shape
    outs[1] = o1
    deconv_0 = deconv2d_audio(input_[:,:inbound1, :, :], outs,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '0', padding=padding)
    
    deconv_1 = deconv2d_audio(input_[:,inbound1:inbound2, :, :], outs,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '1', padding=padding)
    
    outs[1] = o2
    deconv_2 = deconv2d_audio(input_[:,inbound2:, :, :], outs,
             k_h=k_h, k_w=k_w, d_h=d_h, d_w=d_w, stddev=stddev,
             name=name + '2', padding=padding)
    deconv_regions = tf.concat((deconv_0, deconv_1, deconv_2), 1)
    deconv = tf.concat((deconv_all, deconv_regions), 3)
    
    return deconv
    
def deconv2d_audio_local(input_, out_filters,
             k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d"):
    '''
    deconv_11 = conv_local(input_[:,:,0,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '11')
    deconv_12 = conv_local(input_[:,:,1,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '12')
    deconv_21 = conv_local(input_[:,:,0,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '21')
    deconv_22 = conv_local(input_[:,:,1,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '22')
    deconv_1 = deconv_11 + deconv_12
    deconv_2 = deconv_21 + deconv_22
    '''
    #deconv_1 = conv_local(input_[:,:,0,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '1')
    #deconv_2 = conv_local(input_[:,:,1,:], output_shape, k_h, k_w, d_h, d_w, stddev, name + '2')
    input1 = input_[:,:,0,:]
    input2 = input_[:,:,1,:]
    
    '''
    keras.layers.local.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid',
    data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, bias_constraint=None)
    '''
    deconv_11 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input1)
    deconv_12 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input1)
    deconv_21 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input2)
    deconv_22 = Local(filters = out_filters, kernel_size = k_h, strides = d_h)(input2)
    deconv_1 = deconv_11 + deconv_21
    deconv_2 = deconv_12 + deconv_22
    deconv = tf.concat((tf.expand_dims(deconv_1, 2), tf.expand_dims(deconv_2, 2)), 2)

    return deconv
def conv_audio_local(input_, out_filters,
             k,s, stddev=0.02,
             name="deconv2d"):
    input1 = input_[:,:,0,:]
    
    deconv = Local(filters = out_filters, kernel_size = k, strides = s)(input1)

    deconv = tf.expand_dims(deconv, 2)
    return deconv

def conv_local(input_, output_shape, k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,
             name="deconv2d"):
    A_len = output_shape[1]
    F_out = output_shape[-1]
    F_in = input_.get_shape()[-1]
    bs = input_.get_shape()[0].value
    with tf.variable_scope(name):
        act_out = []
        for out_idx in range(A_len):
            act_out_slice = tf.zeros((bs,F_out))
            for in_idx in range(k_h):
                w = tf.get_variable('w_i2n_' + str(out_idx) + '_in_' + str(in_idx), [F_out, F_in],
                            initializer=tf.random_normal_initializer(stddev=stddev))
                w = tf.tile(tf.expand_dims(w, 0), [bs,1,1])
                in_slice = tf.expand_dims(input_[:,out_idx + in_idx,:],-1)
                act = tf.reshape(tf.matmul(w,in_slice), [bs, F_out])
                act_out_slice += act
                
            act_out = act_out + [act_out_slice]
        
        act_out = tf.stack(act_out, 1)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(act_out, biases), act_out.get_shape())
    
        return deconv

def deconvMany(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    deconv_out = [None]*4
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        for idx, val in enumerate([1,3,5,7]):
            w = tf.get_variable('w' + str(idx), [val, val, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[int(o) for o in output_shape],
                                    strides=[1, d_h, d_w, 1])
            biases = tf.get_variable('biases' + str(idx), [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv_out[idx] = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return None
        else:
            return tf.concat(deconv_out,axis=3)

def lrelu(x, leak=0.1, name="lrelu"):
    return tf.maximum(x, leak*x)
def noised(x,std=.2):
    return x + tf.random_normal(x.get_shape(), mean=0,stddev=std)

def noised_gamma(x, std=.2, alpha=.5,beta=1):
    return x + tf.minimum(tf.random_gamma([1], alpha=alpha, beta = beta)[0],2) * \
        tf.random_normal(x.get_shape(), mean=0,stddev=std)

def parametric_relu(_x, name):
    alphas = tf.Variable(tf.ones(_x.get_shape()[-1])*0.0001, name = name)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    
    return pos + neg

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias






