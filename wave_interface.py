# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:32:02 2018

@author: haoqi
"""

import main

#filename = 'data/取样数据1.txt'

def interface(filename = 'data/取样数据1.txt'):
    '''
    function:接口调用，返回参数
    input:原始波形文件路径
    output：
        wave:平均后波形
        f:心率
        loc_peak：峰值位置； y_peak：对应峰值幅度
        loc_valley:波谷位置； y_valley:对应波谷幅值
    '''
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
    '''
    function:接口调用，返回参数Txt文件
    input:原始波形文件路径
    output：
        两个txt文件
        features.txt:保存波形原始特征点，wave,T,loc_peak, y_peak, loc_valley, y_valley
        features_values.txt:保存波形的特征参数
        它们的顺序为：
            wave:平均后波形
            f:心率
            t1_divide_T:主波周期长度与单周期长度比例
            t2_divide_T：潮波波谷与单周期长度比例
            t3_divide_T：潮波波峰与单周期长度比例
            t4_divide_T：重搏波波谷与单周期长度比例
            t5_divide_T：重搏波波峰与单周期长度比例
            H1：主波幅值
            H2_divide_H1：潮波波谷与主波幅值比例
            H3_divide_H1：潮波波峰与主波幅值比例
            H4_divide_H1：重搏波波谷与主波幅值比例
            H5_divide_H1：重搏波波峰与主波幅值比例
            H32_divide_H1：潮波波峰波谷差值与主波幅值比例
            H54_divide_H1：重搏波波峰波谷差值与主波幅值比例
    '''
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
    
    dic = features_dict(wave,loc_peak, y_peak, loc_valley, y_valley)
    dict_values = values(dic)
    
    np.savetxt("features_values.txt", [dict_values[i] for i in dict_values],fmt='%s') #保存为txt文件
    
    return