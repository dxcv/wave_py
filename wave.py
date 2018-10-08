# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:55:11 2018

@author: haoqi
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
#import pandas as pd
from filter import *
from wave_process import *

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
#pb_all, pc_all = load_all_data(filedir = 'data')

'''
np.savetxt("静态压数据.txt", pb_all,fmt='%d') #保存为整型格式
np.savetxt("动态压数据.txt", pc_all,fmt='%d') #保存为整型格式

pb_all = np.loadtxt("静态压数据.txt",dtype=int)
pc_all = np.loadtxt("动态压数据.txt",dtype=int)
'''

def find_peak(a):
    '''
    function:寻找波峰波谷
    input:数据
    outpu：
        count_peak：波峰位置
        y_peak：波峰值
        count_start：起始点位置
        y_start：起始点幅值
    '''
    # 求波峰
    pm = np.amax(a)
    G = 0.6*pm
#    a_large = np.where(a>G, a, 0)
    a_large = np.maximum(a, G)

    a_large_diff = wave_diff(a_large, n=1, axis=-1)
    N = len(a_large_diff)
    
    count_peak = []
    y_peak = []
    for i in range(N-1):
        if a_large_diff[i] >= 0 and a_large_diff[i+1] < 0:
            count_peak.append(i+1)
            y_peak.append(a_large[i+1])
    count_peak = np.asarray(count_peak)
    y_peak = np.asarray(y_peak)
    
    # 求波谷-起始点
    ps = np.min(a)
    G2 = 0.6 * ps
    a_start = np.minimum(a, G2)
    a_start_diff = wave_diff(a_start, n=1, axis=-1)
    
    count_start = []
    y_start = []
    for i in range(N-1):
        if a_start_diff[i] < 0 and a_start_diff[i+1] >= 0:
            count_start.append(i+1)
            y_start.append(a_start[i+1])
    count_start = np.asarray(count_start)
    y_start = np.asarray(y_start)
    
    # 将误识别点去除
    count_start_diff = wave_diff(count_start, n=1, axis=-1)
    need_to_delet = np.where(count_start_diff<100)
    count_start = np.delete(count_start, need_to_delet, axis = 0)
    y_start = np.delete(y_start, need_to_delet, axis = 0)
    return count_peak, y_peak, count_start, y_start

def cutting(a, count_start):
    '''
    function:截取出周期波形
    '''
    cuttings = []
    for i in np.arange(len(count_start)-1):
        cuttings.append(a[count_start[i]:count_start[i+1]])
    return cuttings

def find_features(a):
    '''
    function：寻找特征点
    '''
    
    
#if __name__ == '__main__':
pb_all = np.loadtxt("静态压数据.txt",dtype=int)
pc_all = np.loadtxt("动态压数据.txt",dtype=int)
pbc_all = pb_all+pc_all

aa = pc_all[11]
# 数据处理
aa = normalization(aa)
b,a = signal.butter(3,0.009,'high')

#b, a = signal.iirdesign([0.009, 0.07], [0.007, 0.13], 2, 40)
sf = signal.filtfilt(b,a,aa)

sf1, sf = smooth(sf,n=10)

count_peak, y_peak, count_start, y_start = find_peak(sf)

cuttings = cutting(sf, count_start)

cuttings_org = cutting(aa, count_start)
# 寻找特征点
aa = cuttings[2]
aa_diff = wave_diff(aa)

bb = cuttings_org[2]
bb_diff = wave_diff(bb)

plt.plot(aa)
plt.plot(aa_diff)
#plt.plot(bb)
#plt.plot(bb_diff)
plt.grid(True)

#count_start = count_start[count_start>100]
#plt.subplot(211)
#plt.plot(sf)
#plt.plot(aa)
#plt.subplot(212)

#plt.subplot(212)
    
#plt.figure(1)
#plt.plot(sf)
#plt.scatter(count_peak, y_peak)
#plt.scatter(count_start, y_start)
#plt.plot(aa)




#plt.scatter(count,y)
#plt.plot(sf_large_diff)


#x_sma, x_ema = smooth(sf,n=10)
#aa_x_sma_diff = wave_diff(x_ema)
#
#
##    plt.plot(aa)
#plt.plot(aa_x_sma_diff)

#n = pc_all.shape[0]
#fig = plt.figure()
#for i in np.arange(n):
#    plt.plot(pbc[i,0:1500])
#plt.show()

#a = 0
#pc_filter = np.array(ArithmeticAverage(pc_all[a,0:500],2))
#
#pc_diff = wave_diff(pc_filter,n=1,axis=-1)  #后期需对函数进行优化
#fig = plt.figure()
#plt.subplot(211)
#plt.plot(pc_all[a,0:500])
#plt.subplot(212)
#plt.plot(pc_diff[0:500])
#plt.show()
