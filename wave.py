# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:55:11 2018

@author: haoqi
"""

import numpy as np
import matplotlib.pyplot as plt
import os
#import pandas as pd
from filter import ArithmeticAverage

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
def wave_filter(data):
    '''
    function: 滤波
    '''
    
def wave_diff(data):
    '''
    function：求导
    
    '''

def fitting(data):
    '''
    function: 拟合函数
    
    '''
pb_all = np.loadtxt("静态压数据.txt",dtype=int)
pc_all = np.loadtxt("动态压数据.txt",dtype=int)
pbc_all = pb_all+pc_all

#n = pc_all.shape[0]
#fig = plt.figure()
#for i in np.arange(n):
#    plt.plot(pbc[i,0:1500])
#plt.show()
pc_filter = np.array(ArithmeticAverage(pc_all[0,:],2))

pc_diff = np.diff(pc_filter,n=1,axis=-1)
fig = plt.figure()
plt.subplot(211)
plt.plot(pc_all[0,0:500])
plt.subplot(212)
plt.plot(pc_diff[0:500])
plt.show()



