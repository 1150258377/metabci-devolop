# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal

def bandpower(data, fs, fmin, fmax):
    """计算特定频带的能量
    
    参数:
    data: ndarray, shape (n_channels, n_times)
        输入数据
    fs: float
        采样频率
    fmin: float
        频带下限
    fmax: float
        频带上限
    
    返回:
    ndarray, shape (n_channels,)
        每个通道在指定频带的能量
    """
    n_channels = data.shape[0]
    band_power = np.zeros(n_channels)
    
    for ch in range(n_channels):
        # 计算功率谱密度
        f, pxx = signal.welch(data[ch], fs=fs, nperseg=min(256, data.shape[1]))
        
        # 找到频带范围内的索引
        idx_band = np.logical_and(f >= fmin, f <= fmax)
        
        # 计算频带能量
        band_power[ch] = np.mean(pxx[idx_band])
    
    return band_power 