
import os
import sys
import random
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

import librosa

mod_full_data_files_name = 'D:\\mod_full_data_files_list.txt'

aug_train_data_path = 'D:\\aug_train_data.npz'
aug_test_data_path = 'D:\\aug_test_data.npz'

train_data_for_all = 'D:\\train_data_for_all.npz'

train_data_sampling_one = 'D:\\train_data_sampling_one.npz'
test_data_sampling_one = 'D:\\test_data_sampling_one.npz'

sample_rate = 16000
recording_time = 2
frame_t = 0.025
shift_t = 0.01
buffer_s = 3000

rate_list = [0.97, 0.94, 0.91, 0.88, 1.03, 1.06, 1.09, 1.12, 1.0]

def main():
    generate_train_data_for_all_1(mod_full_data_files_name)

    return

#%%
def time_stretch(data, aug_rate):

    aug_data = librosa.effects.time_stretch(data, aug_rate)
    print(aug_data)

    return aug_data


##
def generate_train_data(text_filepath):

    fwb_train = open(aug_train_data_path, 'wb')
    fwb_test = open(aug_test_data_path, 'wb')

    train_data_list = list()
    test_data_list = list()

    train_label_list = list()
    test_label_list = list()

    with open(text_filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            temp_path = line[0].split('\\\\')

            if temp_path[2] == 'test':
                samplerate, data = wavfile.read(line[0])
                logfb_feat = logfbank(data)
                logfb_feat = util_module.standardization_func(logfb_feat)
                test_data_list.append(logfb_feat)
                test_label_list.append(int(line[-1]))

            elif temp_path[2] == 'train':
                samplerate, data = wavfile.read(line[0])
                for r in rate_list:
                    one_train = list()
                    if r == 1.0:
                        logfb_feat = logfbank(data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)
                    else:
                        # aug_data = time_stretch(data, r)
                        aug_data = librosa.effects.time_stretch(data, r)
                        if len(aug_data) < sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=0.5)

                        elif len(aug_data) > sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=1.0)
                        if len(aug_data) > sample_rate*recording_time:
                            continue

                        logfb_feat = logfbank(aug_data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)

            print('one data complete...')

    random.shuffle(train_data_list)

    temp_train_data = list()
    temp_train_label = list()

    for one in train_data_list:
        # print(one)
        temp_train_data.append(one[0])
        temp_train_label.append(one[1])

    train_data = np.asarray(temp_train_data)
    train_label = np.asarray(temp_train_label)

    test_data = np.asarray(test_data_list)
    test_label = np.asarray(test_label_list)

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=samplerate)
    np.savez_compressed(fwb_test, label=test_label, data=test_data, rate=samplerate)

    print(len(train_data))

    return


##
def generate_train_data_for_all(text_filepath):

    fwb_train = open(train_data_for_all, 'wb')

    train_data_list = list()
    train_label_list = list()

    num = 1
    with open(text_filepath, 'r', encoding='utf-8') as fr:
        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            samplerate, data = wavfile.read(line[0])
            for r in rate_list:
                one_train = list()
                if r == 1.0:
                    logfb_feat = logfbank(data)
                    logfb_feat = util_module.standardization_func(logfb_feat)
                    one_train.append(logfb_feat)
                    one_train.append(int(line[-1]))
                    train_data_list.append(one_train)
                else:
                    # aug_data = time_stretch(data, r)
                    aug_data = librosa.effects.time_stretch(data, r)
                    if len(aug_data) < sample_rate*recording_time:
                        aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                        frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=0.5)

                    elif len(aug_data) > sample_rate*recording_time:
                        aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                        frame_time=frame_t,
                                        shift_time=shift_t,
                                        sample_rate=sample_rate,
                                        buffer_size=buffer_s,
                                        full_size = sample_rate*recording_time,
                                        threshold_value=1.0)
                    if len(aug_data) > sample_rate*recording_time:
                        continue

                    logfb_feat = logfbank(aug_data)
                    logfb_feat = util_module.standardization_func(logfb_feat)
                    one_train.append(logfb_feat)
                    one_train.append(int(line[-1]))
                    train_data_list.append(one_train)

            print('one data complete...', num)
            num+=1

    random.shuffle(train_data_list)

    temp_train_data = list()
    temp_train_label = list()

    for one in train_data_list:
        # print(one)
        temp_train_data.append(one[0])
        temp_train_label.append(one[1])

    train_data = np.asarray(temp_train_data)
    train_label = np.asarray(temp_train_label)

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=samplerate)

    print(len(train_data))

    return


##
def generate_train_data_for_all_1(text_filepath):

    fwb_train = open(train_data_sampling_one, 'wb')
    fwb_test = open(test_data_sampling_one, 'wb')

    train_data_list = list()
    test_data_list = list()

    train_label_list = list()
    test_label_list = list()

    num = 1
    with open(text_filepath, 'r', encoding='utf-8') as fr:
        comm = 'first_c'
        person = 'first_p'

        while True:
            line = fr.readline()
            line = line.split()
            if not line: break

            temp_path = line[0].split('\\\\')

            if temp_path[-2] != comm or temp_path[-3] != person:
                samplerate, data = wavfile.read(line[0])
                logfb_feat = logfbank(data)
                logfb_feat = util_module.standardization_func(logfb_feat)
                test_data_list.append(logfb_feat)
                test_label_list.append(int(line[-1]))
                comm = temp_path[-2]
                person = temp_path[-3]


            else:
                samplerate, data = wavfile.read(line[0])
                for r in rate_list:
                    one_train = list()
                    if r == 1.0:
                        logfb_feat = logfbank(data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)
                    else:
                        # aug_data = time_stretch(data, r)
                        aug_data = librosa.effects.time_stretch(data, r)
                        if len(aug_data) < sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=0.5)

                        elif len(aug_data) > sample_rate*recording_time:
                            aug_data = signal_trigger.evaluate_mean_of_frame(aug_data,
                                            frame_time=frame_t,
                                            shift_time=shift_t,
                                            sample_rate=sample_rate,
                                            buffer_size=buffer_s,
                                            full_size = sample_rate*recording_time,
                                            threshold_value=1.0)
                        if len(aug_data) > sample_rate*recording_time:
                            continue

                        logfb_feat = logfbank(aug_data)
                        logfb_feat = util_module.standardization_func(logfb_feat)
                        one_train.append(logfb_feat)
                        one_train.append(int(line[-1]))
                        train_data_list.append(one_train)

            print('one data complete...', num)
            num+=1

    random.shuffle(train_data_list)

    temp_train_data = list()
    temp_train_label = list()

    for one in train_data_list:
        # print(one)
        temp_train_data.append(one[0])
        temp_train_label.append(one[1])

    train_data = np.asarray(temp_train_data)
    train_label = np.asarray(temp_train_label)

    test_data = np.asarray(test_data_list)
    test_label = np.asarray(test_label_list)

    np.savez_compressed(fwb_train, label=train_label, data=train_data, rate=samplerate)
    np.savez_compressed(fwb_test, label=test_label, data=test_data, rate=samplerate)

    print(len(train_data))

    return








## endl
if __name__ == '__main__':
    main()
