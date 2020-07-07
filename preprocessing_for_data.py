# -*- coding: utf-8 -*-

#%% import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler

#%% standardization
def make_standar(input_numpy):
    
    scaler = StandardScaler()
    scaler.fit(input_numpy)
    result = scaler.transform(input_numpy)
    
    return result



#%% normalization -> max abs scale
def make_MaxAbsScaler(input_numpy):
    transformer = MaxAbsScaler().fit(input_numpy)
    result = transformer.transform(input_numpy)
    
    return result


#%% reshape for cnn
def reshape_for_cnn(input_numpy):
    
    
    return result


