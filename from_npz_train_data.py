# -*- coding: utf-8 -*-

"""
Created on Wed Aug 12 16:53:27 2020

@author: jyback_pnc
"""

import numpy as np
from python_speech_features import mfcc
from python_speech_features import logfbank


class make_train_data:
    
    def __init__(self, **kwarg):
        if "npz_path" in kwarg.keys():
            self.npz_path = kwarg['npz_path']
        if "train_mode" in kwarg.keys():
            self.train_mode = kwarg["train_mode"]


    def write_fb_feat(self):
        
        return


    def make_train_fb_with_mode(self, label, load_data):
        
        return
    
    
    def write_mfcc_feat(self, filename, label, mfcc_data, sample_rate):
        fwb = open(filename, 'wb')
        np.savez_compressed(fwb, label=label, mfcc_data=mfcc_data, rate=sample_rate)
        fwb.close()
        return
    
    
    def make_train_mfcc_with_mode(self, **kwarg):
        
        if "filename" in kwarg.keys():
            filename = kwarg["filename"]

        label, load_data = self.load_train_data()
        
        raw_signal = load_data['data']
        sample_rate = load_data['rate']
        
        mfcc_list = list()
        
        for sig, rate in zip(raw_signal, sample_rate):
            mfcc_feat = mfcc(sig, rate)
            mfcc_list.append(mfcc_feat)
        
        mfcc_data = np.asarray(mfcc_list)
        
        if self.train_mode == 1:
            self.write_mfcc_feat(filename, label, mfcc_data, sample_rate)
            
        elif self.train_mode == 2:
            self.write_mfcc_feat(filename, label, mfcc_data, sample_rate)
        
        return            
    
    
    def write_raw_signal(self, filename, label, data, sample_rate):
        fwb = open(filename, 'wb')
        np.savez_compressed(fwb, label=label, data=data, rate=sample_rate)
        fwb.close()
        return
    
    
    def make_train_raw_sig_with_mode(self, **kwarg):
        
        if "filename" in kwarg.keys():
            filename = kwarg["filename"]
        
        label, load_data = self.load_train_data()
        
        sample_rate = load_data['rate']
        data = load_data['data']
        
        if self.train_mode == 1:
            self.write_raw_signal(filename, label, data, sample_rate)
            
        elif self.train_mode == 2:
            self.write_raw_signal(filename, label, data, sample_rate)
        
        return
    
    
    def make_train_mode_1(self, load_data): # wake up command
        a = load_data['label']
        for i,la in enumerate(a):
            if la != 15:
                a[i] = 0
            else:
                a[i] = 1        
        # print(a)
        return a
    
    
    def make_train_mode_2(self, load_data): # normal command
        a = load_data['label']
        for i,la in enumerate(a):
            if la == 15:
                a[i] = 16
        # print(a)        
        return a
    
    
    def load_train_data(self):
        load_data = np.load(self.npz_path, allow_pickle=True)
        if self.train_mode==1:
            mod_label = self.make_train_mode_1(load_data)
        elif self.train_mode==2:
            mod_label = self.make_train_mode_2(load_data)
            
        return mod_label, load_data
                


if __name__=="__main__":
    print("hello, world~!!")

