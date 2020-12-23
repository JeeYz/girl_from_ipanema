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

# sr, data = wavfile.read(example_filename)
#
# print(data)
#
#
# win_data = signal.hann(len(data))
#
# print(win_data*data)


data = np.random.randint(0, 100, 10)
print(data)
print(type(data))
new = data[1:5]
print(new)
print(type(new))
print(np.mean(new))






## endl
