# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 14:43:44 2018
用于波形信号处理
@author: haoqi
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import butter,lfilter,filtfilt

def normalization(a,norm = False):
    '''
    function:归一化数据
        标准归一化[0,1],均值归一化（变为正态分布）
    '''
    if not norm:
        a = np.asanyarray(a)
        a = (a.T-np.min(a,axis=-1))/(np.max(a,axis=-1)-np.min(a,axis=-1))
    else:
        a = (a.T - a.mean(axis=-1))/a.std(axis=-1)
    a = a.T
    return a


# 平滑
def smooth(x, n=20):
    N = len(x)

    weight = np.ones(n)
    weight /= weight.sum()
    x_sma = np.convolve(x, weight, mode='same')  # 简单移动平均

    weight = np.linspace(1, 0, n)
    weight = np.exp(weight)
    weight /= weight.sum()
    x_ema = np.convolve(x, weight, mode='same')  # 指数移动平均

#    mpl.rcParams['font.sans-serif'] = [u'SimHei']
#    mpl.rcParams['axes.unicode_minus'] = False
#    plt.figure(facecolor='w')
#    plt.plot(np.arange(N), x, 'c-', linewidth=1, label=u'原始')
#    t = np.arange(n - 1, N)
#    plt.plot(t, x_sma, 'g-', linewidth=1, label=u'简单移动平均线')
#    plt.plot(t, x_ema, 'r-', linewidth=1, label=u'指数移动平均线')
#    plt.legend(loc='upper right')
#    plt.grid(True)
#    plt.show()
    return x_sma, x_ema

def kalman(a):
    '''
    function:卡尔曼滤波
    '''
    # 参数初始化
    n_iter = len(a)
    sz = (n_iter,) # size of array
    z = a # observations (normal about x, sigma=0.1)观测值
     
    #Q = 2e-4 # process variance
    Q = np.var(z)
     
    # 分配数组空间
    xhat=np.zeros(sz)      # a posteri estimate of x 滤波估计值
    P=np.zeros(sz)         # a posteri error estimate滤波估计协方差矩阵
    xhatminus=np.zeros(sz) # a priori estimate of x 估计值
    Pminus=np.zeros(sz)    # a priori error estimate估计协方差矩阵
    K=np.zeros(sz)         # gain or blending factor卡尔曼增益
     
    R = 0.08 # estimate of measurement variance, change to see effect
     
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
     
    for k in range(1,n_iter):
        # 预测
        xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
     
        # 更新
        K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    return xhat

def butter_bandpass(lowcut,highcut,fs,order=5):
    nyq = 0.5 * fs #奈奎斯特频率为采样频率的一半
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(order,[low,high],btype = 'band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=5):
    b ,a = butter_bandpass(lowcut,highcut,fs,order = order)
    #y = lfilter(b,a,data)  ##使用IIR或FIR滤波器沿一维过滤数据,b为分子系数向量,a为分母系数向量,data为数据
    y = filtfilt(b,a,data)
    return y  #y为滤波器输出

def fft(a,T,fs):
    f = np.linspace(0, fs, T*fs, endpoint=False)

    ff = np.fft.fft(a)
    ff = np.abs(ff)*2/T/fs
    return f,ff

'''
def ft(x0):
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    n = 20
    N = len(x0)
    x = ((x0 - np.mean(x0)) / (np.max(x0) - np.min(x0))).flat

    # 指数移动平均
    weight = np.linspace(1, 0, n)
    weight = np.exp(weight)
    weight /= weight.sum()
    x_ema = np.convolve(x, weight, mode='valid')
    t = np.arange(n - 1, N)

    # 傅里叶变换(原始数据)
    # plt.subplot(212)
    N2 = len(x)
    w = np.arange(N2) * 2 * np.pi / N2
    f = np.fft.fft(x)
    a = np.abs(f / N2)
    plt.stem(w, a)
    # plt.show()

    # 傅里叶变换(平滑后数据)
    # plt.subplot(212)
#    N2 = len(x_ema)
#    w = np.arange(N2) * 2 * np.pi / N2
#    f = np.fft.fft(x_ema)
#    a = np.abs(f / N2)
#    plt.stem(w, a)

    # 逆变换
    f_real = np.real(f)
    lim = 0.9
    eps = lim * f_real.max()
    f_real[(f_real < eps) & (f_real > -eps)] = 0
    f_imag = np.imag(f)
    eps = lim * f_imag.max()
    f_imag[(f_imag < eps) & (f_imag > -eps)] = 0
    f1 = f_real + f_imag * 1j
    y1 = np.fft.ifft(f1)
    y1 = np.real(y1)

    # plt.subplot(211)
    plt.plot(t, x_ema, label=u'指数移动平均线')
    plt.plot(x, label=u'原始数据')
    plt.plot(y1, label=u'逆变换')

    plt.legend(loc='upper left')
    plt.show()
'''