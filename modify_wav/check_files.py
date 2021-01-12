import os
import sys

from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import wavfile

temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)

from analysis_signal import draw_single_graph, signal_trigger, util_module

train_files_name = 'D:\\train_data_files.txt'
test_files_name = 'D:\\test_data_files.txt'

mod_train_files_name = 'D:\\mod_train_data_files.txt'
mod_test_files_name = 'D:\\mod_test_data_files.txt'

mod_folder_name = 'ASR_train_data_mod'
folder_name = 'ASR_train_data'

## create folder
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

##
def modification_text_file():
    with open(train_files_name, 'r', encoding='utf-8') as fr, \
    open(mod_train_files_name, 'w', encoding='utf-8') as fw:
        while True:
            line = fr.readline()
            if not line: break

            temp = line.split(folder_name)
            new_line = temp[0]+mod_folder_name+temp[1]

            fw.write(new_line)


    with open(test_files_name, 'r', encoding='utf-8') as fr, \
    open(mod_test_files_name, 'w', encoding='utf-8') as fw:
        while True:
            line = fr.readline()
            if not line: break

            temp = line.split(folder_name)
            new_line = temp[0]+mod_folder_name+temp[1]

            fw.write(new_line)

# modification_text_file()


##
def make_data_list():
    full_data_list = list()
    one_file_dict = dict()

    with open(train_files_name, 'r', encoding='utf-8') as f1, \
    open(mod_train_files_name, 'r', encoding='utf-8') as f2:
        while True:
            line = f1.readline()
            line = line.split()
            if not line: break

            line2 = f2.readline()
            line2 = line2.split()

            one_file_dict['file_name'] = line[0]
            one_file_dict['key_label'] = line[1]
            one_file_dict['com_label'] = line[2]
            one_file_dict['mod_file_name'] = line2[0]

            full_data_list.append(one_file_dict)
            one_file_dict = dict()


    with open(test_files_name, 'r', encoding='utf-8') as f1, \
    open(mod_test_files_name, 'r', encoding='utf-8') as f2:
        while True:
            line = f1.readline()
            line = line.split()
            if not line: break

            line2 = f2.readline()
            line2 = line2.split()

            one_file_dict['file_name'] = line[0]
            one_file_dict['key_label'] = line[1]
            one_file_dict['com_label'] = line[2]
            one_file_dict['mod_file_name'] = line2[0]

            full_data_list.append(one_file_dict)
            one_file_dict = dict()

    return full_data_list


full_data_list = make_data_list()
# print(full_data_list)


##
if __name__ == '__main__':
    sample_rate = 16000
    recording_time = 2
    frame_t = 0.025
    shift_t = 0.01
    buffer_s = 3000

    samplerate, data = wavfile.read(full_data_list[0]['file_name'])
    print(samplerate)
    print(data)
    print(len(data))
    draw_single_graph.draw_graph_raw_signal(data, title_name = 'raw signal')
    # plt.show()

    data = util_module.standardization_func(data)
    mod_data = signal_trigger.evaluate_mean_of_frame(data, frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=0.5)
    draw_single_graph.draw_graph_raw_signal(mod_data, title_name = 'modified raw signal')
    plt.show()




















## endl
