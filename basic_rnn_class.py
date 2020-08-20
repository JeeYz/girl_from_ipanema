# -*- coding: utf-8 -*-

from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU


#%% mormal rnn layers
class norm_rnn(layers.Layer):
    def __init__(self, **kwarg):
        super(norm_lstm, self).__init__()
        
        if 'input_mode' in kwarg.keys():
            self.input_flag = kwarg['input_mode']
        if 'rnn_mode' in kwarg.keys():
            self.mode_flag = kwarg['rnn_mode']
        if 'num_of_layers' in kwarg.keys():
            self.num_layers = kwarg['num_of_layers']
            
    
    def transform_input_fcnn_trnn(self, cnn_output):
        
        
        
        return rnn_input
        
    
    def __call__(self, input_val, **kwarg):
        if 'num_of_cells' in kwarg.keys():
            num_cells = kwarg['num_of_cells']
        
        if self.mode_flag == 0:
            output_val = LSTM(self.num_cells)(input_val)
            
        elif self.mode_flag == 1:
            output_val = GRU(self.num_cells)(input_val)
            
        elif self.mode_flag == 2:
            output_val = tf.keras.layers.Bidirectional(LSTM(self.num_cells))(input_val)
            
        elif self.mode_flag == 3:
            output_val = tf.keras.layers.Bidirectional(GRU(self.num_cells))(input_val)
        
        return output_val


#%% __main__
if __name__ == '__main__':
    print('hello, world~!~!')
    



