# -*- coding: utf-8 -*-

#%% explained
'''
class : residual cnn block
class : convolution nn with using resnet
'''


#%% declaration
from tensorflow.keras import layers
import tensorflow as tf


#%% class -> residual cnn block
class residual_cnn_block(layers.Layer):
    
    def __init__(self, **kwarg):
        super(residual_cnn_block, self).__init__()
        
        if "input_shape" in kwarg.keys():
            self.input_shape_conv = kwarg["input_shape"]
        else:
            self.input_shape_conv = None
            
        if "channel_size" in kwarg.keys():
            self.chan_size = kwarg['channel_size']
        if "kernel_size" in kwarg.keys():
            self.ker_size = kwarg['kernel_size']
        
        if "dropout_value" in kwarg.keys():
            dropout_val = kwarg["dropout_value"]
        
        
        # self.conv2d_layer_1 = layers.Conv2D(self.chan_size[0], self.ker_size, 
        #                                     padding='same')
        # self.conv2d_layer_2 = layers.Conv2D(self.chan_size[1], self.ker_size, 
        #                                     padding='same')
            
        
    def __call__(self, init_inputs, inputs, **kwarg):
        conv2d_layer_1 = layers.Conv2D(self.chan_size[0], self.ker_size, 
                                            padding='same')
        conv2d_layer_2 = layers.Conv2D(self.chan_size[1], self.ker_size, 
                                            padding='same')
        
        init_val = inputs
        
        # x = layers.BatchNormalization()(inputs)
        # x = tf.nn.relu(x)
        x = conv2d_layer_1(inputs)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = conv2d_layer_2(x)
        print('*********', self.chan_size)
        y = layers.Conv2D(self.chan_size[1], (1, 1), padding='same')(init_val)
        z = layers.Conv2D(self.chan_size[1], (1, 1), padding='same')(init_inputs)
        x = tf.math.add(y, x)
        x = tf.math.add(x, z)
        x = tf.nn.relu(x)
                
        return x
        

#%% class -> residual net
class residual_net(layers.Layer):
    
    def __init__(self, **kwarg):
        super(residual_net, self).__init__()
        
        if "channels_size" in kwarg.keys():
            self.chan_size = kwarg["channels_size"] # init size
        if "dropout_value" in kwarg.keys():
            self.dropout_val = kwarg["dropout_value"] # init value
        if "num_of_blocks" in kwarg.keys():
            self.num_of_blocks = kwarg["num_of_blocks"]
        if "activation" in kwarg.keys():
            self.activation = kwarg["activation"]
            
        if "input_shape" in kwarg.keys():
            self.input_shape_conv = kwarg["input_shape"]
        else:
            self.input_shape_conv = False
        if "pooling_bool" in kwarg.keys():
            self.pooling_bool = kwarg["pooling_bool"]
        else:
            self.pooling_bool = False
        if "kernel_size" in kwarg.keys():
            self.ker_size = kwarg["kernel_size"]
        if "output_type" in kwarg.keys():
            # 1: 'cnn' / 2: 'rnn' / 3: 'dense' / 4: 'false'
            self.output_type = kwarg['output_type'] 

        
        self.residual_cnn_layer_1 = residual_cnn_block(channel_size=[8, 8], 
                                                kernel_size=self.ker_size)
        self.residual_cnn_layer_2 = residual_cnn_block(channel_size=[16, 16], 
                                                kernel_size=self.ker_size)
        self.residual_cnn_layer_3 = residual_cnn_block(channel_size=[32, 32], 
                                                kernel_size=self.ker_size)
        self.residual_cnn_layer_4 = residual_cnn_block(channel_size=[64, 64], 
                                                kernel_size=self.ker_size)
        self.residual_cnn_layer_5 = residual_cnn_block(channel_size=[128, 128], 
                                                kernel_size=self.ker_size)
        self.residual_cnn_layer_6 = residual_cnn_block(channel_size=[256, 256], 
                                                kernel_size=self.ker_size)
        # self.residual_cnn_layer_7 = residual_cnn_block(channel_size=self.chan_size, 
        #                                         kernel_size=self.ker_size)
        # self.residual_cnn_layer_8 = residual_cnn_block(channel_size=self.chan_size, 
        #                                         kernel_size=self.ker_size)
        # self.residual_cnn_layer_9 = residual_cnn_block(channel_size=self.chan_size, 
        #                                         kernel_size=self.ker_size)
            
        self.pooling_layer = layers.MaxPool2D(pool_size=(4, 1), padding='same')
        
        
    def __call__(self, inputs, **kwarg):
        
        if 'num_of_classes' in kwarg.keys():
            num_class = kwarg['num_of_classes']        
        if 'dense_softmax' in kwarg.keys():
            softmax_bool = kwarg['dense_softmax']
        else:
            softmax_bool = False
        
        init_input = inputs
        
        x = self.residual_cnn_layer_1(init_input, inputs)
        if self.pooling_bool:
            x = self.pooling_layer(x)
        x = self.residual_cnn_layer_2(init_input, x)
        x = self.residual_cnn_layer_3(init_input, x)
        x = self.residual_cnn_layer_4(init_input, x)
        x = self.residual_cnn_layer_5(init_input, x)
        x = self.residual_cnn_layer_6(init_input, x)
        # x = self.residual_cnn_layer_7(init_inputs, x)
        # x = self.residual_cnn_layer_8(init_inputs, x)
        # x = self.residual_cnn_layer_9(init_inputs, x)
        
        
        if softmax_bool:
            x = layers.GlobalAveragePooling2D()(x)
            # x = layers.Flatten()(x)
            # x = layers.Dropout(0.2)(x)
            # x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.1)(x)
            output_val = layers.Dense(num_class, activation='softmax')(x)
        else:
            output_val = x
        
        return output_val


#%% __main__
if __name__ == "__main__":
    print("hello, world~!")

