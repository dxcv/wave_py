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
#import pandas as pd
from filter import *
from wave_process import *
from wave_tools import *
#import pandas as pd

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
        pb = data[:,1:2].T
        pc = data[:,3:].T
        return data,pb,pc
    
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
       
    pb_all = np.zeros([1,6000]).astype(int) #本方法有待改进
    pc_all = np.zeros([1,6000]).astype(int)
    for file in files:
        filename = filedir + '/' + file
        _,pb,pc = loaddata(filename)
        pb_all = np.vstack([pb_all,pb])
        pc_all = np.vstack([pc_all,pc])
    return pb_all[1::], pc_all[1::]
pb_all, pc_all = load_all_data(filedir = 'data')

np.savetxt("静态压数据.txt", pb_all,fmt='%d') #保存为整型格式
np.savetxt("动态压数据.txt", pc_all,fmt='%d') #保存为整型格式

#if __name__ == '__main__':
pb_all = np.loadtxt("静态压数据.txt",dtype=int)
pc_all = np.loadtxt("动态压数据.txt",dtype=int)
pbc_all = pb_all+pc_all


#for i in range(pc_all.shape[0]):
#    pc = pc_all[i]

#*********************
# 验证不同滤波器的影响
#*********************

pc = pc_all[34]

# 数据处理
pc = normalization(pc)

#pc = kalman(pc)

b,a = signal.butter(3,0.005,'high')
sf = signal.filtfilt(b,a,pc)
a1, a = smooth(sf,n=10)
## 寻找特征点

wave = wave_average(a)
wave_length = len(wave)

T = wave_T(fs = 200, length = wave_length)
loc_peak_new, y_peak_new, loc_valley_new, y_valley_new = find_features(wave)

get_figure(wave,loc_peak_new, y_peak_new, loc_valley_new, y_valley_new)




#    plt.plot(wave_dif1,label=u'一阶差分')
#    plt.plot(wave_secdif,label=u'二阶差分')
#    plt.plot(wave_thrdif,label=u'三阶差分')
#plt.scatter(loc_peak1,y_peak1)
#plt.scatter(loc_peak2,y_peak2)
#plt.scatter(loc_peak1[1],wave[loc_peak1[1]])

#plt.savefig('figure/平均波形/fig{}.jpg'.format(i))
