import os
import sys
import random

from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

import numpy as np
import time
from scipy.io import wavfile
from check_files import createFolder, make_data_list

from sklearn.preprocessing import Normalizer

temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

from analysis_signal import draw_single_graph, signal_trigger, util_module
from python_speech_features import logfbank

train_files_name = 'D:\\train_data_files.txt'
test_files_name = 'D:\\test_data_files.txt'

mod_train_files_name = 'D:\\mod_train_data_files.txt'
mod_test_files_name = 'D:\\mod_test_data_files.txt'

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'
mod_full_data_files_name_shuffle = 'D:\\mod_full_data_files_list_shuffle.txt'

mod_train_data_path = 'D:\\aug_train_data.npz'
mod_test_data_path = 'D:\\aug_test_data.npz'

sample_rate = 16000
recording_time = 2
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000



























## endl
