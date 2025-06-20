# -*- coding: utf-8 -*-
"""
DREAMER数据集分析示例
包含数据加载、预处理、特征提取和简单分类
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from metabci.brainda.datasets.dreamer import DREAMER
from metabci.brainda.algorithms.feature_analysis.freq_analysis import bandpower

def main():
    print("开始DREAMER数据集分析...")
    
    # 1. 加载数据集
    dataset = DREAMER()
    print(f"数据集信息:")
    print(f"被试数量: {len(dataset.subjects)}")
    
    # 2. 选择要分析的被试
    subject_id = 1
    print(f"\n处理被试 {subject_id} 的数据...")
    
    try:
        # 获取被试数据
        subject_data = dataset._get_single_subject_data(subject_id)
        
        # 3. 数据预处理
        features = []
        labels = []
        
        for session_name, session_data in subject_data.items():
            for run_name, raw in session_data.items():
                # 带通滤波 (1-45 Hz)
                raw.filter(l_freq=1, h_freq=45, verbose=False)
                
                # 获取EEG数据
                data = raw.get_data()
                
                # 获取事件信息
                events = raw.annotations
                
                # 4. 特征提取
                # 计算各个频带的能量
                bands = {'theta': (4, 8),
                        'alpha': (8, 13),
                        'beta': (13, 30),
                        'gamma': (30, 45)}
                
                for start, duration, description in zip(events.onset, 
                                                      events.duration, 
                                                      events.description):
                    # 提取每个事件的数据片段
                    start_idx = int(start * raw.info['sfreq'])
                    end_idx = start_idx + int(duration * raw.info['sfreq'])
                    epoch = data[:, start_idx:end_idx]
                    
                    # 计算频带能量
                    feature_vector = []
                    for band_name, (fmin, fmax) in bands.items():
                        # 对所有通道计算带通能量
                        band_power = bandpower(epoch, raw.info['sfreq'], fmin, fmax)
                        feature_vector.extend(band_power)
                    
                    features.append(feature_vector)
                    # 将情绪评分转换为二分类标签（高/低）
                    label = 1 if float(description) > 3 else 0
                    labels.append(label)
        
        # 转换为numpy数组
        X = np.array(features)
        y = np.array(labels)
        
        # 5. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 6. 数据标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 7. 训练分类器
        print("\n训练SVM分类器...")
        clf = SVC(kernel='rbf', random_state=42)
        clf.fit(X_train, y_train)
        
        # 8. 评估模型
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n分类结果:")
        print(f"准确率: {accuracy:.3f}")
        print("\n详细分类报告:")
        print(classification_report(y_test, y_pred))
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 