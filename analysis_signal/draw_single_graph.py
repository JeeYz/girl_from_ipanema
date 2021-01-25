#%% import sklearn
import os
import sys
temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

#%% standardization

import numpy as np
import time
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from python_speech_features import logfbank
from analysis_signal.util_module import standardization_func, new_minmax_normal, transpose_the_matrix


#%%
def draw_graph_raw_signal(data, **kwargs):

    title_name = kwargs['title_name']

    plt.figure()
    plt.plot(data)

    plt.xlabel('sample rate')
    plt.ylabel('amplitude')
    plt.title(title_name)

    plt.tight_layout()
    # plt.show()

    return



##
def draw_graph_logfbank(data, sr, **kwargs):
    title_name = kwargs['title_name']

    # data = logfbank(data, sr)
    data = transpose_the_matrix(data)

    plt.figure()
    plt.plot()
    plt.pcolormesh(data)
    # plt.plot(data_0)

    plt.xlabel('frame sequence')
    plt.ylabel('number of filters')
    plt.title(title_name)

    plt.tight_layout()
    plt.colorbar()
    # plt.show()

    return



#%%
def draw_graph_logfbank_norm(data, sr, **kwargs):

    if "title_name" in kwargs.keys():
        title_name = kwargs['title_name']

    data = logfbank(data, sr)
    data = transpose_the_matrix(data)

    data = new_minmax_normal([data])

    plt.figure()
    plt.plot()
    plt.pcolormesh(data[0])
    # plt.plot(data_0)

    plt.xlabel('frame sequence')
    plt.ylabel('number of filters')
    plt.title(title_name)
    plt.colorbar()
    plt.tight_layout()

    return











## endl
