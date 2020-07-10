# -*- coding: utf-8 -*-

#%% import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

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


#%% standard normalization
def make_normalization(input_numpy):
    normalizer = Normalizer().fit(input_numpy)
    result = normalizer.transform(input_numpy)
    
    return result


#%% make normalization by np.linalg
def make_norm_nplinalg(input_numpy):
    result = input_numpy/np.linalg.norm(input_numpy)
    
    return result
    

#%% make normalization with numpy
def make_norm_numpy(input_numpy):
    input_min, input_max = input_numpy.min(), input_numpy.max()
    result = (input_numpy - input_min)/(input_max - input_min)
    
    return result


#%%
def make_stand_numpy(input_numpy):
    result = (input_numpy - np.mean(input_numpy))/np.std(input_numpy)
    
    return result


#%% reshape for cnn #0
def reshape_for_1dcnn(input_numpy, rate):
    result = np.reshape(input_numpy, (len(input_numpy), rate*2, 1))
    
    return result

#%% reshape for cnn #1
def reshape_for_2dcnn(input_numpy):
    result = np.reshape(input_numpy, \
                    (len(input_numpy), len(input_numpy[0]), \
                     len(input_numpy[0][0]), 1))
    
    return result


#%%

