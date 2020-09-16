# -*- coding: utf-8 -*-

#%% explined
'''
class : basic rnn block
class : normal rnn model
option : last layer is dense layer
'''


#%% declaration
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU
import numpy as np


#%% mormal rnn block
class rnn_block(layers.Layer):
    def __init__(self, **kwarg):
        super(rnn_block, self).__init__()       
        
        if 'num_of_cells' in kwarg.keys():
            self.num_cells = kwarg['num_of_cells']
        if 'rnn_mode' in kwarg.keys():
            self.mode_flag = kwarg['rnn_mode']
            

    def __call__(self, input_val, **kwarg):
        
        if self.mode_flag == 'lstm':
            output_val = LSTM(self.num_cells)(input_val)
            
        elif self.mode_flag == 'gru':
            output_val = GRU(self.num_cells)(input_val)
            
        elif self.mode_flag == 'bilstm':
            output_val = tf.keras.layers.Bidirectional(\
                            LSTM(self.num_cells))(input_val)
            
        elif self.mode_flag == 'bigru':
            output_val = tf.keras.layers.Bidirectional(\
                            GRU(self.num_cells))(input_val)
        
        elif self.mode_flag == 'lstm_reseq':
            output_val = LSTM(self.num_cells, return_sequences=True)(input_val)
            
        elif self.mode_flag == 'gru_reseq':
            output_val = GRU(self.num_cells, return_sequences=True)(input_val)
            
        elif self.mode_flag == 'bilstm_reseq':
            output_val = tf.keras.layers.Bidirectional(LSTM(self.num_cells, 
                                        return_sequences=True))(input_val)
            
        elif self.mode_flag == 'bigru_reseq':
            output_val = tf.keras.layers.Bidirectional(GRU(self.num_cells, 
                                        return_sequences=True))(input_val)
        
        return output_val


#%% class normal rnn
class normal_rnn(layers.Layer):
    def __init__(self, **kwarg):
        super(normal_rnn, self).__init__()
        
        if 'num_of_cells' in kwarg.keys():
            self.num_cells = kwarg['num_of_cells']
        if 'cnn_input' in kwarg.keys():
            self.cnn_input = kwarg['cnn_input']
        if 'num_of_layers' in kwarg.keys():
            self.num_layers = kwarg['num_of_layers']
        if 'rnn_mode' in kwarg.keys():
            self.mode_flag = kwarg['rnn_mode']
        if 'num_of_classes' in kwarg.keys():
            self.num_classes = kwarg['num_of_classes']
            
    
    def convert_fcnn_trnn(self, input_tensor):
        # input_tensor = input_tensor.numpy()
        # temp = input_tensor.shape
        # result_numpy = np.reshape(input_tensor, (temp[0], temp[1], -1))
        # return result_numpy
        
        print(input_tensor.shape)
        result_tensor = tf.reshape(input_tensor, [-1, input_tensor.shape[1], \
                                input_tensor.shape[2]*input_tensor.shape[3]])
        print(result_tensor)
        
        return result_tensor
    
    
    def __call__(self, input_data, **kwarg):
        if 'batch_normalization' in kwarg.keys():
            bn_bool = kwarg['batch_normalization']
        if 'num_of_classes' in kwarg.keys():
            num_classes = kwarg['num_of_classes']
        if 'final_rnn_mode' in kwarg.keys():
            fin_mode_flag = kwarg['final_rnn_mode']
            
        if self.cnn_input:
            input_data = self.convert_fcnn_trnn(input_data)
        
        init_rnn_layer = rnn_block(num_of_cells=self.num_cells, 
                                   rnn_mode=self.mode_flag)
        x = init_rnn_layer(input_data)
        
        # if bn_bool:
        #     x = layers.BatchNormalization()(x)        
            
        # rnn_layer = rnn_block(num_of_cells=self.num_cells, 
        #                       rnn_mode=self.mode_flag)
        # for l in range(self.num_layers-1):
        #     x = rnn_layer(x)
        #     if bn_bool:
        #         x = layers.BatchNormalization()(x)
        
        # final_rnn_layer = rnn_block(num_of_cells=self.num_cells, 
        #                       rnn_mode=fin_mode_flag)
        # x = final_rnn_layer(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        answer = layers.Dense(num_classes, activation='softmax')(x)
                
        return answer



#%% __main__
if __name__ == '__main__':
    print('hello, world~!~!')
    



