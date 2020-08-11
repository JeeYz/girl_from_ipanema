# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class basic_cnn1d(tf.keras.layers.Layer):
    
    def __init__(self, num_of_neurons, num_filter, **kwarg):
        super(basic_cnn1d, self).__init__()
        self.num_of_ns = num_of_neurons
        self.size_of_filter = num_filter
        if 'stride' in kwarg.keys():
            self.stride_num = kwarg['stride']
        if 'fil_num' in kwarg.keys():
            self.filter_num = kwarg['fil_num']
        if 'num_ns_cnn' in kwarg.keys():
            self.num_ns_cnn = kwarg['num_ns_cnn']
        
    def __call__(self, **info_of_layers):
        inputs = info_of_layers['input_data']
        input_shape_conv = info_of_layers['input_shape']
        num_dropout_ls = info_of_layers['num_dropout']
        
        for i, _ in enumerate(range(info_of_layers['num_of_ls'])):
            if i == 0:
                x = layers.Conv1D(self.num_of_ns, 
                                self.size_of_filter, 
                                strides = self.stride_num,
                                padding='same',
                                activation='relu',
                                input_shape=input_shape_conv)(inputs)
            else:
                if i == num_dropout_ls:
                    x = layers.Dropout(0.5)(x)    
                x = layers.Conv1D(self.num_ns_cnn,
                                self.filter_num, 
                                padding='same',
                                activation='linear',
                                input_shape=input_shape_conv)(x)
        
        x = layers.Flatten()(x)
        
        if info_of_layers['dense_ls'] == True:
            x = layers.Dense(512, activation='linear')(x)
            
        return x


class basic_cnn2d(tf.keras.layers.Layer):
    
    def __init__(self, num_of_neurons, num_filter):
        super(basic_cnn2d, self).__init__()
        self.num_of_ns = num_of_neurons
        self.size_of_filter = num_filter
        
    def __call__(self, **info_of_layers):
        inputs = info_of_layers['input_data']
        input_shape_conv = info_of_layers['input_shape']
        num_dropout_ls = info_of_layers['num_dropout']
        
        for i, _ in enumerate(range(info_of_layers['num_of_ls'])):
            if i == 0:
                x = tf.keras.layers.Conv2D(self.num_of_ns, 
                                           self.size_of_filter, 
                                           padding='same',
                                           activation='linear',
                                           input_shape=input_shape_conv)(inputs)
            else:
                if i == num_dropout_ls:
                    x = layers.Dropout(0.5)(x)    
                x = tf.keras.layers.Conv2D(self.num_of_ns, 
                                           self.size_of_filter, 
                                           padding='same',
                                           activation='linear',
                                           input_shape=input_shape_conv)(x)
        
        x = layers.Flatten()(x)
        
        if info_of_layers['dense_ls'] == True:
            x = layers.Dense(512, activation='linear')(x)
            
        return x

    

if __name__ == '__main__':
    print('hello, DL world~!!')
    

