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


class basic_cnn2d(layers.Layer):
    
    def __init__(self, **kwarg):
        super(basic_cnn2d, self).__init__()
        
        if "channels_list" in kwarg.keys():
            self.chan_list = kwarg["channels_list"]
        if "kernel_size" in kwarg.keys():
            self.ker_size = kwarg["kernel_size"]
        if "dropout_list" in kwarg.keys():
            self.dropout_list = kwarg["dropout_list"]
        if "activation" in kwarg.keys():
            self.activiation = kwarg["activation"]
        if "padding" in kwarg.keys():
            self.padding = kwarg["padding"]
        if "num_of_layers" in kwarg.keys():
            self.num_of_layers = kwarg["num_of_layers"] - 1
                
        
    def __call__(self, inputs):
        
        x = layers.Conv2D(self.chan_list[0], self.ker_size,
            padding='same', activation='relu', input_shape=input_shape_conv)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(self.chan_list[0], self.ker_size,
                          padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        for i in range(self.num_of_layers):
            x = layers.Conv2D(self.chan_list[0],
                              self.ker_size,
                              padding='same',
                              activation='relu')(inputs)
        
        return outputs
    
    
    
    def __call__(self, **info_of_layers):
        inputs = info_of_layers['input_data']
        input_shape_conv = info_of_layers['input_shape']
        num_dropout_ls = info_of_layers['num_dropout']
        
        for i, _ in enumerate(range(info_of_layers['num_of_ls'])):
            if i == 0:
                x = layers.Conv2D(self.num_of_ns, 
                                    self.size_of_filter, 
                                    padding='same',
                                    activation='linear',
                                    input_shape=input_shape_conv)(inputs)
            else:
                if i == num_dropout_ls:
                    x = layers.Dropout(0.5)(x)    
                x = layers.Conv2D(self.num_of_ns, 
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
    

