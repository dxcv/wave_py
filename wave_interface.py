# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:32:02 2018

@author: haoqi
"""

from main import *

#filename = 'data/取样数据1.txt'

def interface(filename = 'data/取样数据1.txt'):
    '''接口调用'''
    
    pc = loaddata(filename)
    pc = normalization(pc)
    
    b,a = signal.butter(3,0.009,'high')
    sf = signal.filtfilt(b,a,pc)
    b,a = signal.butter(3,0.1,'low')
    pc1 = signal.filtfilt(b,a,sf)
    
    wave = wave_average(pc1)
    
    wave_length = len(wave)
    
    T = wave_T(fs = 200, length = wave_length)
    loc_peak, y_peak, loc_valley, y_valley = find_features(wave)
    
    loc_peak, y_peak, loc_valley, y_valley = features_choose(wave,loc_peak, y_peak, loc_valley, y_valley)
    
    #get_figure(wave,loc_peak, y_peak, loc_valley, y_valley)
    
    return wave, T, loc_peak, y_peak, loc_valley, y_valley

def interface_txt(filename = 'data/取样数据1.txt'):
    pc = loaddata(filename)
    pc = normalization(pc)
    
    b,a = signal.butter(3,0.009,'high')
    sf = signal.filtfilt(b,a,pc)
    b,a = signal.butter(3,0.1,'low')
    pc1 = signal.filtfilt(b,a,sf)
    
    wave = wave_average(pc1)
    
    wave_length = len(wave)
    
    T = wave_T(fs = 200, length = wave_length)
    loc_peak, y_peak, loc_valley, y_valley = find_features(wave)
    
    loc_peak, y_peak, loc_valley, y_valley = features_choose(wave,loc_peak, y_peak, loc_valley, y_valley)
    get_figure(wave,loc_peak, y_peak, loc_valley, y_valley)
    
    features = [wave,T,loc_peak, y_peak, loc_valley, y_valley]
    
    np.savetxt("features.txt", features,fmt='%s') #保存为txt文件
    
    return

def interface1(filename = 'data/取样数据1.txt'):
    pc = loaddata(filename)
    pc = normalization(pc)
    
    b,a = signal.butter(3,0.009,'high')
    sf = signal.filtfilt(b,a,pc)
    b,a = signal.butter(3,0.1,'low')
    pc1 = signal.filtfilt(b,a,sf)
    
    wave = wave_average(pc1)
    
    wave_length = len(wave)
    
    T = wave_T(fs = 200, length = wave_length)
    loc_peak, y_peak, loc_valley, y_valley = find_features(wave)
    
    loc_peak, y_peak, loc_valley, y_valley = features_choose(wave,loc_peak, y_peak, loc_valley, y_valley)
    get_figure(wave,loc_peak, y_peak, loc_valley, y_valley)
    
    features = [wave,T,loc_peak, y_peak, loc_valley, y_valley]