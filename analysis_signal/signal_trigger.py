import sys
sys.path.append('D:\\')
sys.path.append('..\\')

from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn import preprocessing

from python_speech_features import logfbank
from files_locations import example_filename
from draw_single_graph import draw_graph_logfbank, draw_graph_logfbank_norm, draw_graph_raw_signal
import draw_single_graph
import draw_multi_graphs

from util_module import standardization_func, new_minmax_normal, transpose_the_matrix

trigger_val = 0.5


def evaluate_mean_of_frame(data, **kwargs):

    if "frame_time" in kwargs.keys():
        frame_time = kwargs['frame_time']
    if "shift_time" in kwargs.keys():
        shift_time = kwargs['shift_time']
    if "sample_rate" in kwargs.keys():
        sr = kwargs['sample_rate']

    frame_size = int(sr*frame_time)
    shift_size = int(sr*shift_time)

    num_frames = len(data)//(frame_size-shift_size)+1

    mean_val_list = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
            one_frame_data = data[temp_n:temp_n+frame_size]
        mean_val_list.append(np.mean(np.abs(one_frame_data)))

    for i,start in enumerate(mean_val_list):
        if trigger_val < start:
            start_index = i
            break

    for i,end in enumerate(reversed(mean_val_list)):
        if trigger_val < end:
            end_index = len(mean_val_list)-i
            break

    return data[(frame_size-shift_size)*start_index:(frame_size-shift_size)*end_index]
