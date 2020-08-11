# -*- coding: utf-8 -*-

import os


class make_files_list_asr:
    
    def __init__(self, **kwarg):
        
        if "path" in kwarg.keys():
            self.path = kwarg["path"]
        if "file_format" in kwarg.keys():
            self.file_format = kwarg["file_format"]
        if "return_file_path" in kwarg.keys():
            self.make_file_path = kwarg["return_file_path"]
    
        self.return_files_list = []
        
        
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
    a = make_files_list_asr(path="d:\\ASR_audio_files",
                            file_format=".wav",
                            return_file_path="c:\\Users\\jyback_pnc\\Desktop")
    
    a.make_file_process()
    
    # a.find_target_files()
    
    # for el in a.return_files_list:
    #     print(el)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    