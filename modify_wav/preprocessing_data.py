import os
import sys

from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

import numpy as np
import time
from scipy.io import wavfile
from check_files import createFolder, make_data_list

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

sample_rate = 16000
recording_time = 2
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000


##
def main():
    # full_data_list = make_data_list()
    # wav_to_data(full_data_list)



    return


##
def wav_to_data(data_list):

    for one in data_list:
        one_name = one['file_name']
        mod_one_name = one['mod_file_name']
        samplerate, data = wavfile.read(one_name)
        data = util_module.standardization_func(data)
        mod_data = signal_trigger.evaluate_mean_of_frame(data, frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=0.5)

        mod_directory = mod_one_name.split('\\\\')
        mod_dir_name = '\\\\'.join(mod_directory[:-1])
        createFolder(mod_dir_name)
        # wavfile.write(mod_one_name, samplerate, mod_data)

        if len(mod_data) > sample_rate*recording_time:

            mod_data = signal_trigger.evaluate_mean_of_frame(data, frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=1.0)

            # draw_single_graph.draw_graph_raw_signal(mod_data, title_name='raw signal')

            if len(mod_data) > sample_rate*recording_time:
                print(mod_one_name)
                print(len(mod_data))
                draw_single_graph.draw_graph_raw_signal(mod_data, title_name='raw signal')
                continue

        wavfile.write(mod_one_name, samplerate, mod_data)

    plt.show()

    return


##
def generate_train_data():


    return


















## endl
if __name__ == '__main__':
    main()
