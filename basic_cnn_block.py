# -*- coding: utf-8 -*-

#%% declaration
from tensorflow.keras import layers


#%% class -> cnn_block
class cnn_block(layers.Layer):
    
    def __init__(self, **kwarg):
        super(cnn_block, self).__init__()
        
        if "activation" in kwarg.keys():
            self.activation = kwarg["activation"]
            
        if "input_shape" in kwarg.keys():
            self.input_shape_conv = kwarg["input_shape"]
        else:
            self.input_shape_conv = []
            
        
    def __call__(self, inputs, **kwarg):
        
        if "kernel_size" in kwarg.keys():
            ker_size = kwarg["kernel_size"]
        if "dropout_value" in kwarg.keys():
            dropout_val = kwarg["dropout_value"]
        if "pooling_bool" in kwarg.keys():
            pooling_bool = kwarg["pooling_bool"]
        if "channel_size" in kwarg.keys():
            chan_size = kwarg["channel_size"]
        
        if self.input_shape_conv != []:
            x = layers.Conv2D(chan_size, ker_size, padding='same', 
                              activation=self.activation, 
                              input_shape=self.input_shape_conv)(inputs)
        else:
            x = layers.Conv2D(chan_size, ker_size, padding='same', 
                              activation=self.activation)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(chan_size, ker_size, padding='same', 
                          activation=self.activation)(x)
        x = layers.BatchNormalization()(x)
        if pooling_bool == True:
            x = layers.MaxPooling2D(pool_size=(3, 1), padding='same')(x)
            # x = layers.MaxPooling2D(pool_size=(3, 1), strides=(2, 1),
            #                       padding='same')(x)
            # x = layers.MaxPooling2D(pool_size=(4, 1), strides=(3, 1),
            #                         padding='same')(x)
        output_ts = layers.Dropout(dropout_val)(x)
                
        return output_ts
        

#%% class -> norm_cnn
class norm_cnn(layers.Layer):
    
    def __init__(self, **kwarg):
        super(norm_cnn, self).__init__()
        
        if "channels_size" in kwarg.keys():
            self.chan_size = kwarg["channels_size"] # init size
        if "dropout_value" in kwarg.keys():
            self.dropout_val = kwarg["dropout_value"] # init value
        if "num_of_blocks" in kwarg.keys():
            self.num_of_blocks = kwarg["num_of_blocks"]
        
        self.chan_list = []
        for i in range(self.num_of_blocks):
            self.chan_size = self.chan_size*2
            self.chan_list.append(self.chan_size)
        
        self.dropout_list = []
        for i in range(self.num_of_blocks):
            self.dropout_val = self.dropout_val + 0.1
            self.dropout_list.append(self.dropout_val)
        
        if "activation" in kwarg.keys():
            self.activation = kwarg["activation"]
        if "input_shape" in kwarg.keys():
            self.input_shape_conv = kwarg["input_shape"]
        if "pooling_bool" in kwarg.keys():
            self.pooling_bool = kwarg["pooling_bool"]
        if "kernel_size" in kwarg.keys():
            self.ker_size = kwarg["kernel_size"]
                
        
    def __call__(self, inputs, num_class):
        
        cnn_block_layer_init = cnn_block(activation=self.activation, 
                                          input_shape=self.input_shape_conv)
        
        x = cnn_block_layer_init(inputs, channel_size=self.chan_list[0],
                                 kernel_size=self.ker_size, pooling_bool=self.pooling_bool,
                                 dropout_value=self.dropout_list[0])
        
        cnn_block_layer = cnn_block(activation=self.activation)
        for i in range(self.num_of_blocks-1):
            x = cnn_block_layer(x, channel_size=self.chan_list[i+1],
                                 kernel_size=self.ker_size, pooling_bool=self.pooling_bool,
                                 dropout_value=self.dropout_list[i+1])
        
        x = cnn_block_layer(x, channel_size=512,
                            kernel_size=(3,3), pooling_bool=False,
                            dropout_value=0.3)        
        
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Flatten()(x)
        # x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='linear')(x)
        # x = layers.Dropout(0.3)(x)
        output_val = layers.Dense(num_class, activation='softmax')(x)
        
        return output_val


#%% __main__
if __name__ == "__main__":
    print("hello, world~!")

