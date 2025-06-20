# -*- coding: utf-8 -*-
"""
DREAMER dataset for emotion recognition.
"""
from typing import Union, Optional, Dict, List, cast
from pathlib import Path
import os

import numpy as np
import mne
from mne.io import Raw, RawArray
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.channels import upper_ch_names
from ..utils.io import loadmat
import mat73

class DREAMER(BaseDataset):
    """DREAMER dataset for emotion recognition.

    The DREAMER dataset contains EEG recordings from 23 subjects while watching
    emotional video clips. The dataset includes both EEG signals and self-reported
    emotional responses.

    References
    ----------
    .. [1] Katsigiannis, S., & Ramzan, N. (2018). DREAMER: A database for emotion
           recognition through EEG and ECG signals from wireless low-cost
           off-the-shelf devices. IEEE journal of biomedical and health informatics,
           22(1), 98-107.
    """

    def __init__(self,
                 dataset_code: str = 'DREAMER',
                 subjects: Union[List[int], int, None] = None,
                 events: Optional[List[str]] = None,
                 channels: Optional[List[str]] = None,
                 srate: float = 128.0):
        self.dataset_code = dataset_code
        self.srate = srate
        
        # 默认包含数据集中所有23名被试
        if subjects is None:
            self.subjects = list(range(1, 24))  # DREAMER共有23名被试，编号1-23
        else:
            if isinstance(subjects, int):
                self.subjects = [subjects]
            else:
                self.subjects = subjects
        # 确保事件默认包含全部情绪维度
        if events is None:
            self.events = list(self._EVENTS.keys())
        else:
            self.events = events
        # 通道默认设置
        if channels is None:
            self.channels = self._CHANNELS
        else:
            self.channels = channels

    @property
    def _EVENTS(self) -> Dict[str, int]:
        """Return the event codes for emotion ratings."""
        return {
            'valence_low': 1,
            'valence_high': 2,
            'arousal_low': 3,
            'arousal_high': 4,
            'dominance_low': 5,
            'dominance_high': 6
        }
    
    @property
    def _CHANNELS(self) -> List[str]:
        """Return the list of channel names."""
        return [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8',
            'FC6', 'F4', 'F8', 'AF4'
        ]

    def data_path(self, subject: int) -> str:
        """Return the path to the data file for the given subject.
        
        Args:
            subject (int): Subject number
            
        Returns:
            str: Path to the data file
        """
        # DREAMER.mat文件在项目根目录
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        return os.path.join(base_dir, 'DREAMER.mat')

    def _get_single_subject_data(self, subject: int) -> Dict:
        """Return the data for a single subject.
        
        Args:
            subject (int): Subject number
            
        Returns:
            dict: Dictionary containing the data for the subject
        """
        data_path = self.data_path(subject)
        
        # 加载DREAMER.mat文件
        try:
            data = mat73.loadmat(data_path)
        except:
            # 如果mat73加载失败，尝试使用scipy的loadmat
            data = loadmat(data_path)
        
        if 'DREAMER' not in data:
            raise ValueError("Invalid DREAMER data format")
            
        # DREAMER字段是一个mat_struct或dict
        dreamer_struct = data['DREAMER']
        # 支持两种结构: dict 或 mat_struct
        if isinstance(dreamer_struct, dict):
            data_array = dreamer_struct['Data']
        else:
            data_array = dreamer_struct.Data

        subject_data = data_array[subject - 1]  # 索引从0开始
        
        # 根据结构类型选择访问方式
        def _get_attr(obj, field):
            """Helper to fetch attribute/field from mat_struct or dict."""
            if isinstance(obj, dict):
                return obj[field]
            else:
                return getattr(obj, field)

        # 取出EEG刺激段和情绪评分数组
        eeg_struct = _get_attr(subject_data, 'EEG')
        eeg_stimuli = _get_attr(eeg_struct, 'stimuli')  # ndarray of shape (18, n_times, n_channels)

        valence_arr = _get_attr(subject_data, 'ScoreValence')
        arousal_arr = _get_attr(subject_data, 'ScoreArousal')
        dominance_arr = _get_attr(subject_data, 'ScoreDominance')

        sessions = {}
        for video_idx in range(len(eeg_stimuli)):
            eeg_data = eeg_stimuli[video_idx]  # shape (n_times, n_channels)

            # 获取情绪评分
            valence = float(valence_arr[video_idx])
            arousal = float(arousal_arr[video_idx])
            dominance = float(dominance_arr[video_idx])
            
            # 创建MNE raw对象
            ch_names = self._CHANNELS
            ch_types = ['eeg'] * len(ch_names)
            info = mne.create_info(ch_names=ch_names, sfreq=self.srate, ch_types=ch_types)
            
            # 转换数据格式
            eeg_data = np.array(eeg_data).T  # 转置为 (n_channels, n_times)
            raw = mne.io.RawArray(eeg_data, info)
            
            # 添加事件标注
            onset = 0  # 视频开始时间
            duration = len(eeg_data[0]) / self.srate  # 视频持续时间
            
            # 创建事件标注
            annotations = mne.Annotations(
                onset=[onset],
                duration=[duration],
                description=[str(valence)]  # 使用valence作为标签
            )
            raw.set_annotations(annotations)
            
            # 存储数据
            session_name = f'session_{video_idx + 1}'
            sessions[session_name] = {'run_1': raw}
            
        return sessions 