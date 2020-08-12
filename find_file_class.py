# -*- coding: utf-8 -*-

import os
from scipy.io import wavfile
import numpy as np
import time


class make_files_list_asr:
    
    def __init__(self, **kwarg):
        
        if "path" in kwarg.keys():
            self.path = kwarg["path"]
        if "file_format" in kwarg.keys():
            self.file_format = kwarg["file_format"]
        if "return_file_path" in kwarg.keys():
            self.make_file_path = kwarg["return_file_path"]
        if "label_file_path" in kwarg.keys():
            temp = kwarg["label_file_path"]
        
        self.return_files_list = []
        self.label_dict = dict()
        self.make_label_dict(temp)
        
        
    def make_label_dict(self, t_path): # t_path is label file's path
        none_list = ['hibixbi', 'hiplus', 'okgoogle']
        label_num = 0
        with open(t_path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if not line:break
                line = line.split()
                
                if line[0] not in none_list:
                    self.label_dict[line[0]] = label_num
                    label_num += 1 
    
        
    def read_wav_file(self):
        result_filename = 'result.txt'
        result_binary = 'result_numpy.npz'
        temp = self.make_file_path +'\\'+ result_filename
        temp_1 = self.make_file_path +'\\'+ result_binary
        data_list = list()
        label_list = list()
        with open(temp, 'r', encoding='utf-8') as fr,\
        open(temp_1, 'wb') as fwb:
            while True:
                line = fr.readline()
                line = line.split('\n')[0]
                if not line: break
                # print(line)
                fs, data = wavfile.read(line)
                # print(fs)
                # print(data)
                tline = line.split('\\')
                if tline[-2] in self.label_dict.keys():
                    # print(self.label_dict[tline[-2]])
                    a = np.array([self.label_dict[tline[-2]]])
                elif tline[-2] == 'hipnc2':
                    # print(self.label_dict['hipnc'])
                    a = np.array([self.label_dict['hipnc']])
                else:
                    # print(self.label_dict['none'])
                    a = np.array([self.label_dict['none']])
                
                data_list.append(data)
                label_list.append(a)
                # print(result)
                # print(len(result))
            data_list = np.asarray(data_list)
            label_list = np.asarray(label_list)
            np.savez_compressed(fwb, label=label_list, data=data_list)
                
        return
    
        
    def make_file_process(self):        
        self.find_target_files()
        self.make_files_list_file()
        
        
    def make_files_list_file(self):
        result_filename = 'result.txt'
        temp = self.make_file_path +'\\'+ result_filename
        with open(temp, 'w', encoding='utf-8') as wf:
            for one_file in self.return_files_list:
                wf.write(one_file)
                wf.write("\n")
        
        return
        
    
    def find_target_files(self):
        self.return_files_list = []
        for (path, dir, files) in os.walk(self.path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == self.file_format:
                    self.return_files_list.append(path+'\\'+filename)

    
    def return_directories_name(self):
        
        Target_list = os.listdir(self.path)
        Target_list_data = [file for file in Target_list if file.endswith(self.file_format)]
        
        return Target_list_data



if __name__ == '__main__':
    print("hello, world~!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    