# -*- coding: utf-8 -*-

#%% explanation
'''
auto encoder class...
'''

#%% declaration
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU
import numpy as np


#%% class auto encoder
class AutoEncoder(layers.Layer):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoding_layer = ae_Encoder()
        self.decoding_layer = ae_Decoder()
        
    def __call__(self, input_x):
        x = self.encoding_layer(input_x)
        res = self.decoding_layer(x)
        
        return res


#%% encoder
class ae_Encoder(layers.Layer):
    def __init__(self):
        super(ae_Encoder, self).__init__()
        self.conv2D_layer_1 = layers.conv2D(32, (3, 3), padding='same')
        self.conv2D_layer_2 = layers.conv2D(16, (3, 3), padding='same')
        self.conv2D_layer_3 = layers.conv2D(8, (3, 3), padding='same')
        self.pooling_layer = layers.MaxPooling2D(pool_size=(10, 1), padding='same')
        
    def __call__(self, input_x):
        x = self.conv2D_layer_1(input_x)
        x = self.pooling_layer(x)
        x = self.conv2D_layer_2(x)
        x = self.pooling_layer(x)
        x = self.conv2D_layer_3(x)
        x = self.pooling_layer(x)
        
        return x
        
        
#%% decoder
class ae_Decoder(layers.Layer):
    def __init__(self):
        super(ae_Decoder, self).__init__()
        self.conv2D_layer_1 = layers.conv2D(8, (3, 3), padding='same')
        self.conv2D_layer_2 = layers.conv2D(16, (3, 3), padding='same')
        self.conv2D_layer_3 = layers.conv2D(32, (3, 3), padding='same')
        self.upsampling_layer = layers.UpSampling2D(pool_size=(10, 1), padding='same')
    
    def __call__(self, input_x):
        x = self.conv2D_layer_1(input_x)
        x = self.upsampling_layer(x)
        x = self.conv2D_layer_2(x)
        x = self.upsampling_layer(x)
        x = self.conv2D_layer_3(x)
        x = self.upsampling_layer(x)
        
        return x


#%% __main__
if __name__ == '__main__':
    print('hello, world~!')