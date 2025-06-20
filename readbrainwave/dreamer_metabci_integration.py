# -*- coding: utf-8 -*-
"""
DREAMER数据集与MetaBCI框架集成
将DREAMER.mat文件转换为MetaBCI可用的格式，并使用MetaBCI的特征提取功能
"""

import scipy.io
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import mne
from mne.io import RawArray
from mne import create_info

# MetaBCI imports
from metabci.brainda.algorithms.decomposition.csp import CSP, FBCSP
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.feature_analysis.freq_analysis import FrequencyAnalysis
from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import TimeFrequencyAnalysis
from metabci.brainda.algorithms.utils.model_selection import EnhancedLeaveOneGroupOut
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.paradigms import MotorImagery
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class DREAMERDataset(BaseDataset):
    """
    DREAMER数据集类，继承自MetaBCI的BaseDataset
    """
    
    def __init__(self, mat_file_path: str):
        """
        初始化DREAMER数据集
        
        Parameters
        ----------
        mat_file_path : str
            DREAMER.mat文件的路径
        """
        self.mat_file_path = mat_file_path
        
        # 定义事件类型（基于情感评分）
        self._EVENTS = {
            "high_valence": (1, (0, 10)),      # 高愉悦度
            "low_valence": (2, (0, 10)),       # 低愉悦度
            "high_arousal": (3, (0, 10)),      # 高唤醒度
            "low_arousal": (4, (0, 10)),       # 低唤醒度
            "high_dominance": (5, (0, 10)),    # 高支配度
            "low_dominance": (6, (0, 10)),     # 低支配度
        }
        
        # DREAMER数据集的电极通道
        self._CHANNELS = [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9",
            "PO7", "PO3", "O1", "IZ", "OZ", "POZ", "PZ", "CPZ", "FPZ", "FP2", "AF8", "AF4",
            "AFZ", "FZ", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCZ", "CZ",
            "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10",
            "PO8", "PO4", "O2"
        ]
        
        # 加载数据获取被试信息
        self._load_dreamer_data()
        
        super().__init__(
            dataset_code="dreamer",
            subjects=list(range(1, self.no_of_subjects + 1)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=self.eeg_sampling_rate,
            paradigm="emotion"  # 情感识别范式
        )
    
    def _load_dreamer_data(self):
        """加载DREAMER.mat文件并提取基本信息"""
        print(f"正在加载DREAMER数据: {self.mat_file_path}")
        mat_data = scipy.io.loadmat(self.mat_file_path)
        
        # 获取DREAMER数据
        dreamer_data = mat_data['DREAMER'][0, 0]
        
        # 获取主要数据
        data = dreamer_data['Data'][0, 0]
        self.eeg_sampling_rate = dreamer_data['EEG_SamplingRate'][0, 0]
        self.ecg_sampling_rate = dreamer_data['ECG_SamplingRate'][0, 0]
        self.eeg_electrodes = dreamer_data['EEG_Electrodes'][0, 0]
        self.no_of_subjects = dreamer_data['noOfSubjects'][0, 0]
        self.no_of_video_sequences = dreamer_data['noOfVideoSequences'][0, 0]
        
        # 提取评分数据
        self.valence_scores = [int(v[0]) if isinstance(v, np.ndarray) else int(v) for v in data['ScoreValence'][0]]
        self.arousal_scores = [int(a[0]) if isinstance(a, np.ndarray) else int(a) for a in data['ScoreArousal'][0]]
        self.dominance_scores = [int(d[0]) if isinstance(d, np.ndarray) else int(d) for d in data['ScoreDominance'][0]]
        
        # 存储原始数据
        self.eeg_data = data['EEG'][0, 0]
        self.ecg_data = data['ECG'][0, 0]
        
        print(f"数据集信息:")
        print(f"  被试数量: {self.no_of_subjects}")
        print(f"  视频序列数量: {self.no_of_video_sequences}")
        print(f"  EEG采样率: {self.eeg_sampling_rate}")
        print(f"  ECG采样率: {self.ecg_sampling_rate}")
        print(f"  EEG电极数量: {self.eeg_electrodes}")
    
    def data_path(self, subject: Union[str, int], path: Optional[Union[str, Path]] = None,
                  force_update: bool = False, update_path: Optional[bool] = None,
                  proxies: Optional[Dict[str, str]] = None,
                  verbose: Optional[Union[bool, str, int]] = None) -> List[List[Union[str, Path]]]:
        """返回数据路径（对于本地文件，直接返回mat文件路径）"""
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject id: {subject}")
        
        return [[self.mat_file_path]]
    
    def _get_single_subject_data(self, subject: Union[str, int],
                                verbose: Optional[Union[bool, str, int]] = None) -> Dict[str, Dict[str, Raw]]:
        """获取单个被试的数据"""
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject id: {subject}")
        
        subject_idx = subject - 1  # 转换为0基索引
        
        # 获取被试的EEG数据
        subject_eeg = self.eeg_data[subject_idx]
        baseline_data = subject_eeg['baseline'].item()
        stimuli_data = subject_eeg['stimuli'].item()
        
        # 创建MNE Raw对象
        raw_data = {}
        
        # 处理baseline数据
        for trial_idx, trial_data in enumerate(baseline_data):
            if isinstance(trial_data, np.ndarray) and trial_data.size > 0:
                # 创建MNE信息对象
                info = create_info(
                    ch_names=self._CHANNELS,
                    sfreq=self.eeg_sampling_rate,
                    ch_types=['eeg'] * len(self._CHANNELS)
                )
                
                # 创建Raw对象
                raw = RawArray(trial_data.T, info)
                raw_data[f"baseline_trial_{trial_idx+1}"] = raw
        
        # 处理stimuli数据
        for trial_idx, trial_data in enumerate(stimuli_data):
            if isinstance(trial_data, np.ndarray) and trial_data.size > 0:
                # 创建MNE信息对象
                info = create_info(
                    ch_names=self._CHANNELS,
                    sfreq=self.eeg_sampling_rate,
                    ch_types=['eeg'] * len(self._CHANNELS)
                )
                
                # 创建Raw对象
                raw = RawArray(trial_data.T, info)
                raw_data[f"stimuli_trial_{trial_idx+1}"] = raw
        
        return {"session_1": raw_data}


class DREAMERFeatureExtractor:
    """
    DREAMER数据集特征提取器
    使用MetaBCI的特征提取算法
    """
    
    def __init__(self, dataset: DREAMERDataset):
        self.dataset = dataset
        self.srate = dataset.eeg_sampling_rate
    
    def extract_csp_features(self, X: np.ndarray, y: np.ndarray, n_components: int = 4) -> np.ndarray:
        """
        使用CSP提取空间特征
        
        Parameters
        ----------
        X : np.ndarray
            EEG数据，形状 (n_trials, n_channels, n_samples)
        y : np.ndarray
            标签
        n_components : int
            CSP组件数量
            
        Returns
        -------
        np.ndarray
            CSP特征
        """
        # 创建CSP模型
        csp = CSP(n_components=n_components)
        
        # 训练CSP
        csp.fit(X, y)
        
        # 提取特征
        features = csp.transform(X)
        
        return features
    
    def extract_fbcsp_features(self, X: np.ndarray, y: np.ndarray, n_components: int = 4) -> np.ndarray:
        """
        使用FBCSP提取特征
        
        Parameters
        ----------
        X : np.ndarray
            EEG数据，形状 (n_trials, n_channels, n_samples)
        y : np.ndarray
            标签
        n_components : int
            CSP组件数量
            
        Returns
        -------
        np.ndarray
            FBCSP特征
        """
        # 生成滤波器组
        wp = [(4, 8), (8, 12), (12, 30), (30, 45)]
        ws = [(2, 10), (6, 14), (10, 32), (25, 50)]
        filterbank = generate_filterbank(wp, ws, srate=self.srate, order=4, rp=0.5)
        
        # 创建FBCSP模型
        fbcsp = FBCSP(
            n_components=n_components,
            n_mutualinfo_components=4,
            filterbank=filterbank
        )
        
        # 训练FBCSP
        fbcsp.fit(X, y)
        
        # 提取特征
        features = fbcsp.transform(X)
        
        return features
    
    def extract_frequency_features(self, X: np.ndarray, meta: pd.DataFrame, event: str) -> Dict:
        """
        提取频域特征
        
        Parameters
        ----------
        X : np.ndarray
            EEG数据
        meta : pd.DataFrame
            元数据
        event : str
            事件类型
            
        Returns
        -------
        Dict
            频域特征
        """
        # 创建频域分析对象
        freq_analyzer = FrequencyAnalysis(X, meta, event=event, srate=self.srate)
        
        # 计算平均信号
        mean_data = freq_analyzer.stacking_average(data=[], _axis=0)
        
        # 计算功率谱密度
        features = {}
        for ch_idx, ch_name in enumerate(self.dataset._CHANNELS[:10]):  # 只取前10个通道
            f, psd = freq_analyzer.power_spectrum_periodogram(mean_data[ch_idx])
            features[f"psd_{ch_name}"] = psd
            features[f"freq_{ch_name}"] = f
        
        return features
    
    def extract_time_frequency_features(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        提取时频特征
        
        Parameters
        ----------
        X : np.ndarray
            EEG数据
        y : np.ndarray
            标签
            
        Returns
        -------
        Dict
            时频特征
        """
        # 创建时频分析对象
        tf_analyzer = TimeFrequencyAnalysis(self.srate)
        
        features = {}
        
        # 对每个类别进行分析
        for class_label in np.unique(y):
            class_data = X[y == class_label]
            mean_data = np.mean(class_data, axis=0)
            
            # 短时傅里叶变换
            nfft = mean_data.shape[1]
            f, t, Zxx = tf_analyzer.fun_stft(mean_data, nperseg=1000, axis=1, nfft=nfft)
            
            # 存储时频特征
            features[f"stft_class_{class_label}"] = {
                'frequencies': f,
                'times': t,
                'spectrogram': Zxx
            }
        
        return features


class DREAMEREmotionClassifier:
    """
    DREAMER情感分类器
    """
    
    def __init__(self, dataset: DREAMERDataset):
        self.dataset = dataset
        self.feature_extractor = DREAMERFeatureExtractor(dataset)
        self.models = {}
    
    def prepare_data(self, subjects: List[int] = None, emotion_type: str = 'valence') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        准备训练数据
        
        Parameters
        ----------
        subjects : List[int]
            被试列表
        emotion_type : str
            情感类型 ('valence', 'arousal', 'dominance')
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]
            X, y, meta
        """
        if subjects is None:
            subjects = self.dataset.subjects[:5]  # 默认使用前5个被试
        
        X_list = []
        y_list = []
        meta_list = []
        
        for subject in subjects:
            subject_idx = subject - 1
            
            # 获取被试的EEG数据和评分
            subject_eeg = self.dataset.eeg_data[subject_idx]
            stimuli_data = subject_eeg['stimuli'].item()
            
            # 获取情感评分
            if emotion_type == 'valence':
                scores = self.dataset.valence_scores[subject_idx]
            elif emotion_type == 'arousal':
                scores = self.dataset.arousal_scores[subject_idx]
            elif emotion_type == 'dominance':
                scores = self.dataset.dominance_scores[subject_idx]
            else:
                raise ValueError(f"Unknown emotion type: {emotion_type}")
            
            # 处理每个trial
            for trial_idx, trial_data in enumerate(stimuli_data):
                if isinstance(trial_data, np.ndarray) and trial_data.size > 0:
                    # 确保数据形状正确
                    if trial_data.shape[0] == len(self.dataset._CHANNELS):
                        X_list.append(trial_data)
                        
                        # 根据评分创建标签（二分类：高/低）
                        score = scores[trial_idx] if isinstance(scores, list) else scores
                        label = 1 if score > 5 else 0  # 阈值设为5
                        y_list.append(label)
                        
                        # 创建元数据
                        meta = pd.DataFrame({
                            'subject': [subject],
                            'trial': [trial_idx + 1],
                            'emotion_type': [emotion_type],
                            'score': [score],
                            'label': [label]
                        })
                        meta_list.append(meta)
        
        if not X_list:
            raise ValueError("No valid data found")
        
        X = np.array(X_list)
        y = np.array(y_list)
        meta = pd.concat(meta_list, ignore_index=True)
        
        return X, y, meta
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray, method: str = 'csp') -> object:
        """
        训练分类器
        
        Parameters
        ----------
        X : np.ndarray
            特征数据
        y : np.ndarray
            标签
        method : str
            特征提取方法 ('csp', 'fbcsp')
            
        Returns
        -------
        object
            训练好的分类器
        """
        if method == 'csp':
            features = self.feature_extractor.extract_csp_features(X, y)
        elif method == 'fbcsp':
            features = self.feature_extractor.extract_fbcsp_features(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 创建分类器管道
        classifier = make_pipeline(SVC(kernel='rbf', probability=True))
        
        # 训练分类器
        classifier.fit(features, y)
        
        return classifier
    
    def evaluate_classifier(self, X: np.ndarray, y: np.ndarray, classifier: object, method: str = 'csp') -> Dict:
        """
        评估分类器性能
        
        Parameters
        ----------
        X : np.ndarray
            测试数据
        y : np.ndarray
            真实标签
        classifier : object
            训练好的分类器
        method : str
            特征提取方法
            
        Returns
        -------
        Dict
            评估结果
        """
        if method == 'csp':
            features = self.feature_extractor.extract_csp_features(X, y)
        elif method == 'fbcsp':
            features = self.feature_extractor.extract_fbcsp_features(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 预测
        y_pred = classifier.predict(features)
        y_prob = classifier.predict_proba(features)
        
        # 计算准确率
        accuracy = accuracy_score(y, y_pred)
        
        # 分类报告
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_prob,
            'report': report
        }


def main():
    """主函数：演示DREAMER数据集与MetaBCI的集成"""
    
    # 设置DREAMER.mat文件路径
    mat_file_path = "DREAMER.mat"  # 请根据实际路径修改
    
    if not os.path.exists(mat_file_path):
        print(f"错误：找不到文件 {mat_file_path}")
        print("请确保DREAMER.mat文件在当前目录或指定正确的路径")
        return
    
    try:
        # 创建DREAMER数据集对象
        print("正在初始化DREAMER数据集...")
        dreamer_dataset = DREAMERDataset(mat_file_path)
        
        # 创建情感分类器
        print("正在创建情感分类器...")
        classifier = DREAMEREmotionClassifier(dreamer_dataset)
        
        # 准备数据（使用前3个被试进行演示）
        print("正在准备数据...")
        subjects = [1, 2, 3]
        
        # 对三种情感类型进行分类
        emotion_types = ['valence', 'arousal', 'dominance']
        methods = ['csp', 'fbcsp']
        
        results = {}
        
        for emotion_type in emotion_types:
            print(f"\n正在处理 {emotion_type} 情感分类...")
            
            # 准备数据
            X, y, meta = classifier.prepare_data(subjects, emotion_type)
            print(f"数据形状: X={X.shape}, y={y.shape}")
            print(f"标签分布: {np.bincount(y)}")
            
            # 使用留一法交叉验证
            spliter = EnhancedLeaveOneGroupOut(return_validate=False)
            accuracies = []
            
            for train_ind, test_ind in spliter.split(X, y=y, groups=meta['subject']):
                X_train, y_train = X[train_ind], y[train_ind]
                X_test, y_test = X[test_ind], y[test_ind]
                
                # 训练分类器
                model = classifier.train_classifier(X_train, y_train, method='csp')
                
                # 评估分类器
                result = classifier.evaluate_classifier(X_test, y_test, model, method='csp')
                accuracies.append(result['accuracy'])
            
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            print(f"{emotion_type} 分类结果:")
            print(f"  平均准确率: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
            
            results[emotion_type] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'accuracies': accuracies
            }
        
        # 显示总体结果
        print("\n" + "="*50)
        print("总体分类结果:")
        print("="*50)
        for emotion_type, result in results.items():
            print(f"{emotion_type:12s}: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
        
        # 特征提取演示
        print("\n正在演示特征提取...")
        X, y, meta = classifier.prepare_data([1], 'valence')
        
        # 频域特征
        print("提取频域特征...")
        freq_features = classifier.feature_extractor.extract_frequency_features(X, meta, 'high_valence')
        print(f"频域特征数量: {len(freq_features)}")
        
        # 时频特征
        print("提取时频特征...")
        tf_features = classifier.feature_extractor.extract_time_frequency_features(X, y)
        print(f"时频特征数量: {len(tf_features)}")
        
        print("\nDREAMER数据集与MetaBCI集成完成！")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 