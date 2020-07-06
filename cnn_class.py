# -*- coding: utf-8 -*-


import tensorflow as tf


class basic_cnn(tf.keras.layers):
    
    def __init__(self, num_of_neurons):
        super(basic_cnn, self).__init__()
        self.num_of_ns = num_of_neurons
        
    def call(self, input_data, **info_of_layers):
        
        pass
    
    



if __name__ == '__main__':
    print('hello, DL world~!!')
    

