import sys
sys.path.append('D:\\')
temp = __file__.split('\\')
temp = '\\'.join(temp[:-2])
sys.path.append(temp)
import os

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn import preprocessing
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import librosa
import soundfile
import time

from python_speech_features import logfbank
from files_locations import example_filename
from draw_single_graph import draw_graph_raw_signal, draw_graph_logfbank
from util_module import standardization_func, new_minmax_normal, transpose_the_matrix
from signal_trigger import evaluate_mean_of_frame

from modify_wav import check_files

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'
zeroth_files_list = 'D:\\zeroth_files_list.txt'
zeroth_data_path = 'D:\\voice_data_backup\\zeroth_korean.tar\\zeroth_korean'
zeroth_none_data_path = 'D:\\voice_data_backup\\zeroth_none'
full_new_train_data_with_none = 'D:\\full_new_train_data_with_none.txt'

sample_rate = 16000
threshold = 0.5
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000
recording_time = 2


#%%
def find_target_files(fpath, file_format):

    files_list = list()

    for (path, dir, files) in os.walk(fpath):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == file_format:
                files_list.append(path+'\\'+filename)

    return files_list


##
def write_wav_files(files_list):

    for one in files_list:
        data, sr = soundfile.read(one)
        if sr != 16000:
            print(sr, one)
        else:
            temp_path = one.split('.')
            temp_path[-1] = 'wav'
            temp_path = '.'.join(temp_path)
            # print(temp_path)
            soundfile.write(temp_path, data, sr, format='wav', subtype='PCM_16')

    return


##
def add_buffer(data, len_data):

    noise_data = np.random.randn(buffer_s-len_data)*0.01
    result = list()
    result.extend(noise_data)
    result.extend(data)

    return result


##
def devide_data(data):

    data_list = list()

    add_num = 5
    at_least_len = 7000
    max_len = 20000

    frame_size = int(sample_rate*0.025)
    shift_size = int(sample_rate*0.01)

    num_frames = len(data)//(frame_size-shift_size)+1

    data = standardization_func(data)

    temp = list()
    temp_list = list()
    low_temp = list()

    for i in range(num_frames):
        temp_n = i*(frame_size-shift_size)
        if temp_n+frame_size > len(data):
            one_frame_data = data[temp_n:len(data)]
        else:
            one_frame_data = data[temp_n:temp_n+frame_size-shift_size]

            if threshold < np.mean(np.abs(one_frame_data)):
                if low_temp != [] and len(low_temp) < add_num*(frame_size-shift_size):
                    temp.extend(low_temp)

                temp.extend(one_frame_data)
                low_temp = list()
            else:
                if temp != [] and len(low_temp) > add_num*(frame_size-shift_size):
                    st_num = temp_n-len(temp)-(add_num-1)*(frame_size-shift_size)
                    end_num = temp_n+frame_size-shift_size-len(temp)
                    front_data = data[st_num:end_num]

                    new_list = list()
                    new_list.extend(front_data)
                    new_list.extend(temp)
                    temp = add_buffer(new_list, add_num*(frame_size-shift_size))
                    temp.extend(low_temp)
                    data_list.append(temp)
                    temp = list()
                    # print(len(temp))

                low_temp.extend(one_frame_data)
                # print(len(low_temp))


    # print(len(data_list))

    new_data_list = list()

    for one in data_list:
        if len(one) > at_least_len and len(one) < max_len:
            new_data_list.append(one)

    # print(data_list)
    # for one in new_data_list:
    #     draw_graph_raw_signal(one, title_name='raw')
    #     print(len(one))
    #
    # draw_graph_raw_signal(data, title_name='raw')
    #
    # plt.show()
    # time.sleep(10000)

    return new_data_list


##
def refine_data(data_list):
    result_list = list()
    for one in data_list:
        refine_d = evaluate_mean_of_frame(one,
                        frame_time=frame_t,
                        shift_time=shift_t,
                        sample_rate=sample_rate,
                        buffer_size=buffer_s,
                        full_size = sample_rate*recording_time,
                        threshold_value=threshold)

        result_list.append(refine_d)
        if len(refine_d) != 32000:
            print(one)
            print(len(refine_d))


    #     draw_graph_raw_signal(refine_d, title_name='raw')
    #
    # plt.show()
    # time.sleep(10000)

    return result_list


##
def generate_wav_files(data_list, target_path):
    sr = 16000
    for i,one in enumerate(data_list):
        num = str(i)
        soundfile.write(target_path+'_'+num+'.wav', one, sr, format='wav', subtype='PCM_16')

    return


##
def make_none_data(file_path, target_path):
    sr, data = wavfile.read(file_path)
    # print(data)
    # print(file_path, sr)
    # data = standardization_func(data)
    # draw_graph_raw_signal(data, title_name='raw')
    # plt.show()
    # time.sleep(1000)

    result = devide_data(data)
    result = refine_data(result)
    generate_wav_files(result, target_path)

    return


##
def main():

    # flac_list = find_target_files(zeroth_data_path, '.flac')
    # write_wav_files(flac_list)

    zeroth_list = find_target_files(zeroth_data_path, '.wav')

    for i,one in enumerate(zeroth_list):
        # if i == 3000:
        #     break
        temp_path = one.split('zeroth_korean')
        temp_fn = temp_path[-1].split('\\')[-1].split('.')[-2]
        temp = temp_path[-1].split('\\')[:-1]
        temp = '\\'.join(temp)

        folder_path = zeroth_none_data_path + '\\' + temp
        check_files.createFolder(folder_path)

        file_path = folder_path + '\\' + temp_fn
        make_none_data(one, file_path)
        print('\rone file is done....\t%d' %i, end='')

    # draw_graph_raw_signal(data, title_name='raw')
    # plt.show()

    return
















##
if __name__ == '__main__':
    main()
