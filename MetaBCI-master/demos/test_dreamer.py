# -*- coding: utf-8 -*-
"""
Test script for DREAMER dataset loading and basic processing.
"""
import os
import sys
import numpy as np
import mne

# 添加父目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from metabci.brainda.datasets.dreamer import DREAMER

def main():
    print("开始测试DREAMER数据集加载...")
    
    # 初始化DREAMER数据集
    dataset = DREAMER()
    print(f"数据集信息:")
    print(f"  被试数量: {len(dataset.subjects)}")
    print(f"  通道数量: {len(dataset.channels)}")
    print(f"  采样率: {dataset.srate} Hz")
    print(f"  事件类型: {list(dataset.events.keys())}")
    
    # 加载第一个被试的数据
    subject = 1
    print(f"\n加载被试 {subject} 的数据...")
    
    try:
        # 获取被试数据
        subject_data = dataset._get_single_subject_data(subject)
        
        # 打印数据信息
        print("\n数据组织:")
        for session_name, session_data in subject_data.items():
            print(f"\n{session_name}:")
            for run_name, raw in session_data.items():
                print(f"  {run_name}:")
                print(f"    数据形状: {raw.get_data().shape}")
                print(f"    通道: {raw.ch_names}")
                print(f"    采样率: {raw.info['sfreq']} Hz")
                
                # 显示事件信息
                events = mne.find_events(raw)
                if len(events) > 0:
                    print(f"    事件数量: {len(events)}")
                    print(f"    事件类型: {np.unique(events[:, 2])}")
                
                # 基本预处理示例
                print("\n    执行基本预处理...")
                # 带通滤波
                raw.filter(l_freq=1, h_freq=40, verbose=False)
                print("    完成带通滤波 (1-40 Hz)")
                
                # 重采样（如果需要）
                if raw.info['sfreq'] > 128:
                    raw.resample(128, verbose=False)
                    print("    完成重采样 (128 Hz)")
                
                # 显示预处理后的数据形状
                print(f"    预处理后数据形状: {raw.get_data().shape}")
                
                # 只显示第一个trial的信息
                break
        
        print("\n数据加载和预处理测试完成！")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 