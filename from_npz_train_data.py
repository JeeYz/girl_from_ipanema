# -*- coding: utf-8 -*-

"""
Created on Wed Aug 12 16:53:27 2020

@author: jyback_pnc
"""

import numpy as np


class make_train_data:
    def __init__(self, **kwarg):
        if "npz_path" in kwarg.keys():
            self.npz_path = kwarg['npz_path']
        if "train_mode" in kwarg.keys():
            self.train_mode = kwarg["train_mode"]


    def change_mfcc(self):
        
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
            if la == 16:
                a[i] = 15
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

