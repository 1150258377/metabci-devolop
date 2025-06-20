import scipy.io
import numpy as np
import pandas as pd
import os

def flatten_structured_array(arr, prefix=""):
    """
    递归展平结构化数组
    """
    if arr.dtype.names is None:
        return {prefix: arr}
    
    result = {}
    for name in arr.dtype.names:
        field = arr[name]
        new_prefix = f"{prefix}_{name}" if prefix else name
        
        if field.dtype.names is not None:
            # 递归处理嵌套结构
            nested = flatten_structured_array(field, new_prefix)
            result.update(nested)
        else:
            result[new_prefix] = field
    
    return result

def extract_eeg_data(eeg_array):
    """
    提取EEG数据并转换为DataFrame
    """
    all_data = []
    for i, subject_data in enumerate(eeg_array):
        # subject_data 是结构体，直接访问字段
        baseline_data = subject_data['baseline'].item()
        stimuli_data = subject_data['stimuli'].item()
        # 处理baseline数据
        for j, trial_data in enumerate(baseline_data):
            if isinstance(trial_data, np.ndarray) and trial_data.size > 0:
                df_baseline = pd.DataFrame(trial_data)
                df_baseline['subject'] = i + 1
                df_baseline['trial'] = j + 1
                df_baseline['condition'] = 'baseline'
                all_data.append(df_baseline)
        # 处理stimuli数据
        for j, trial_data in enumerate(stimuli_data):
            if isinstance(trial_data, np.ndarray) and trial_data.size > 0:
                df_stimuli = pd.DataFrame(trial_data)
                df_stimuli['subject'] = i + 1
                df_stimuli['trial'] = j + 1
                df_stimuli['condition'] = 'stimuli'
                all_data.append(df_stimuli)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def extract_ecg_data(ecg_array):
    """
    提取ECG数据并转换为DataFrame
    """
    all_data = []
    for i, subject_data in enumerate(ecg_array):
        baseline_data = subject_data['baseline'].item()
        stimuli_data = subject_data['stimuli'].item()
        # 处理baseline数据
        for j, trial_data in enumerate(baseline_data):
            if (
                isinstance(trial_data, np.ndarray)
                and trial_data.size > 0
                and len(trial_data.shape) == 2
                and trial_data.shape[1] == 2
            ):
                df_baseline = pd.DataFrame(trial_data, columns=['ECG1', 'ECG2'])
                df_baseline['subject'] = i + 1
                df_baseline['trial'] = j + 1
                df_baseline['condition'] = 'baseline'
                all_data.append(df_baseline)
        # 处理stimuli数据
        for j, trial_data in enumerate(stimuli_data):
            if (
                isinstance(trial_data, np.ndarray)
                and trial_data.size > 0
                and len(trial_data.shape) == 2
                and trial_data.shape[1] == 2
            ):
                df_stimuli = pd.DataFrame(trial_data, columns=['ECG1', 'ECG2'])
                df_stimuli['subject'] = i + 1
                df_stimuli['trial'] = j + 1
                df_stimuli['condition'] = 'stimuli'
                all_data.append(df_stimuli)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def convert_dreamer_to_dataframes(mat_file_path):
    """
    将DREAMER.mat文件转换为多个DataFrame
    """
    print(f"正在读取文件: {mat_file_path}")
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # 获取DREAMER数据
    dreamer_data = mat_data['DREAMER'][0, 0]
    
    # 获取主要数据
    data = dreamer_data['Data'][0, 0]
    eeg_sampling_rate = dreamer_data['EEG_SamplingRate'][0, 0]
    ecg_sampling_rate = dreamer_data['ECG_SamplingRate'][0, 0]
    eeg_electrodes = dreamer_data['EEG_Electrodes'][0, 0]
    no_of_subjects = dreamer_data['noOfSubjects'][0, 0]
    no_of_video_sequences = dreamer_data['noOfVideoSequences'][0, 0]
    
    print(f"被试数量: {no_of_subjects}")
    print(f"视频序列数量: {no_of_video_sequences}")
    print(f"EEG采样率: {eeg_sampling_rate}")
    print(f"ECG采样率: {ecg_sampling_rate}")
    print(f"EEG电极: {eeg_electrodes}")
    
    # 提取基本信息
    age = [int(a[0]) if isinstance(a, np.ndarray) else int(a) for a in data['Age'][0]]
    gender = [str(g[0]) if isinstance(g, np.ndarray) else str(g) for g in data['Gender'][0]]
    valence_scores = [int(v[0]) if isinstance(v, np.ndarray) else int(v) for v in data['ScoreValence'][0]]
    arousal_scores = [int(a[0]) if isinstance(a, np.ndarray) else int(a) for a in data['ScoreArousal'][0]]
    dominance_scores = [int(d[0]) if isinstance(d, np.ndarray) else int(d) for d in data['ScoreDominance'][0]]
    eeg_data = data['EEG'][0, 0]
    ecg_data = data['ECG'][0, 0]
    
    # 创建被试信息DataFrame
    subjects_df = pd.DataFrame({
        'subject_id': range(1, len(age) + 1),
        'age': age,
        'gender': gender,
        'valence_score': valence_scores,
        'arousal_score': arousal_scores,
        'dominance_score': dominance_scores
    })
    
    print(subjects_df.head())
    print(f"subjects_df shape: {subjects_df.shape}")
    
    # 提取EEG数据
    print("正在处理EEG数据...")
    eeg_df = extract_eeg_data(eeg_data)
    
    # 提取ECG数据
    print("正在处理ECG数据...")
    ecg_df = extract_ecg_data(ecg_data)
    
    # 保存为CSV文件
    subjects_df.to_csv('dreamer_subjects.csv', index=False)
    print("被试信息已保存为 dreamer_subjects.csv")
    
    if not eeg_df.empty:
        eeg_df.to_csv('dreamer_eeg_data.csv', index=False)
        print("EEG数据已保存为 dreamer_eeg_data.csv")
    
    if not ecg_df.empty:
        ecg_df.to_csv('dreamer_ecg_data.csv', index=False)
        print("ECG数据已保存为 dreamer_ecg_data.csv")
    
    # 保存元数据
    metadata = {
        'eeg_sampling_rate': eeg_sampling_rate,
        'ecg_sampling_rate': ecg_sampling_rate,
        'no_of_subjects': no_of_subjects,
        'no_of_video_sequences': no_of_video_sequences,
        'eeg_electrodes': str(eeg_electrodes)
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv('dreamer_metadata.csv', index=False)
    print("元数据已保存为 dreamer_metadata.csv")
    
    return {
        'subjects': subjects_df,
        'eeg': eeg_df,
        'ecg': ecg_df,
        'metadata': metadata_df
    }

def main():
    file_path = r"C:\Users\jing pengqiang\Downloads\DREAMER.mat"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        dataframes = convert_dreamer_to_dataframes(file_path)
        
        print("\n数据转换完成！")
        print(f"被试数量: {len(dataframes['subjects'])}")
        print(f"EEG数据行数: {len(dataframes['eeg'])}")
        print(f"ECG数据行数: {len(dataframes['ecg'])}")
        
        # 显示被试信息的前几行
        print("\n被试信息预览:")
        print(dataframes['subjects'].head())
        
        if not dataframes['eeg'].empty:
            print("\nEEG数据预览:")
            print(dataframes['eeg'].head())
        
        if not dataframes['ecg'].empty:
            print("\nECG数据预览:")
            print(dataframes['ecg'].head())
            
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 