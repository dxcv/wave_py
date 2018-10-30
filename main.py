# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:55:11 2018

@author: haoqi
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy import signal
from filter import *
from wave_process import *
from wave_tools import *

# 参数设定
fs = 200    #采样率
T = 30  #采样时间

def loaddata(filename):
    ''' 
    function : 加载脉象数据
    input : 
        filename : 文件名，filename = 'data/取样数据1.txt'
    output : 
        pb : 静态压
        pc : 脉象信号
    '''
    array_list = []
    with open(filename) as f:
        data_original = f.read().splitlines()
        for i in data_original:
            array_line = i.split(',')
            array_list.append(array_line)
        #转为int32数据
        data = np.array(array_list)
        data = data.astype(int)
        pb = data[:,1].T
        pc = data[:,3].T
        
        #pb = data[:,1:2].T #如果是把多个文件一起生成一个数据文件，需要改成此种形式
        #pc = data[:,3:].T
        #return data,pb,pc
        
        return pc

def load_all_data(filedir = 'data'):
    '''
    function ： 通过子文件加载所有数据
    input :
        filedir : 目录
    output :
        pb_all : 静态压所有数据
        pc_all : 动态压所有数据
    '''
    files = os.listdir(filedir)
    pb_all = np.zeros([1,6000]).astype(int)
    pc_all = np.zeros([1,6000]).astype(int)
    for file in files:
        filename = filedir + '/' + file
        _,pb,pc = loaddata(filename)
        pb_all = np.vstack([pb_all,pb])
        pc_all = np.vstack([pc_all,pc])
    return pb_all[1::], pc_all[1::]

#pb_all, pc_all = load_all_data(filedir = '脉象数据20181024')

#np.savetxt("静态压数据.txt", pb_all,fmt='%d') #保存为整型格式
#np.savetxt("动态压数据.txt", pc_all,fmt='%d') #保存为整型格式


#pb_all = np.loadtxt("静态压数据.txt",dtype=int)
#pc_all = np.loadtxt("动态压数据.txt",dtype=int)
#pbc_all = pb_all+pc_all
#for i in pc_all:
#    pc = i

if __name__ == '__main__':
    
    filename = '脉象数据20181024/取样数据01R.txt'
    pc = loaddata(filename)
    #pc = pc_all[30]
    pc = normalization(pc)
    
    b,a = signal.butter(3,0.009,'high')
    sf = signal.filtfilt(b,a,pc)
    b,a = signal.butter(3,0.1,'low')
    pc1 = signal.filtfilt(b,a,sf)
    
    wave = wave_average(pc1)
    
    wave_length = len(wave)
    
    T = wave_T(fs = 200, length = wave_length)
    loc_peak_new, y_peak_new, loc_valley_new, y_valley_new = find_features(wave)
    
    loc_peak_new, y_peak_new, loc_valley_new, y_valley_new = features_choose(wave,loc_peak_new, y_peak_new, loc_valley_new, y_valley_new)
    
    get_figure(wave,loc_peak_new, y_peak_new, loc_valley_new, y_valley_new)

