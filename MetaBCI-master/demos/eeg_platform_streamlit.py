# -*- coding: utf-8 -*-
"""
Streamlit EEG demo with interactive NN visualization.
"""
import streamlit as st
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from metabci.brainda.datasets.dreamer import DREAMER
import mne
import scipy.io as sio
import mat73

# ------------------------------------------------------------------
# Helper plotting functions (must appear BEFORE main logic)
# ------------------------------------------------------------------

def plot_timeseries(data, fs, title, channel_idx=0):
    t = np.arange(data.shape[1]) / fs
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t, data[channel_idx] * 1e6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


def plot_psd(data, fs, title, channel_idx=0):
    f, pxx = signal.welch(data[channel_idx], fs=fs, nperseg=1024)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.semilogy(f, pxx)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# ------------------------------------------------------------------
# Dummy EEG generator
# ------------------------------------------------------------------

def generate_dummy_eeg(duration_s: float, fs: int, n_channels: int = 14):
    n_times = int(duration_s * fs)
    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_times) * 1e-6
    t = np.arange(n_times) / fs
    data += 1e-5 * np.sin(2 * np.pi * 10 * t)  # add 10 Hz tone
    return data

# ------------------------------------------------------------------
# Pre-processing utilities
# ------------------------------------------------------------------

def bandpass_filter(data: np.ndarray, fs: int, l_freq: float, h_freq: float):
    """Band-pass filter with safe fallback for very short signals."""
    b, a = signal.butter(4, [l_freq, h_freq], fs=fs, btype="band")
    # filtfilt 需要输入长度 > padlen (一般 ≈ 3*max(len(a),len(b)))
    padlen = 3 * max(len(a), len(b))
    if data.shape[-1] > padlen:
        return signal.filtfilt(b, a, data, axis=-1)
    # 否则退化为 lfilter（单向），以避免报错
    return signal.lfilter(b, a, data, axis=-1)


def optional_integrate(data: np.ndarray):
    return np.cumsum(data, axis=-1)

# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def simple_feature(data: np.ndarray):
    return np.mean(data ** 2, axis=-1)  # power

# ------------------------------------------------------------------
# Simple 2-layer MLP
# ------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden1), nn.ReLU(), nn.Dropout(dropout)]
        if hidden2 > 0:
            layers += [nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden2, out_dim)]
        else:
            layers += [nn.Linear(hidden1, out_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# ================================================================
# Streamlit UI
# ================================================================

st.title("EEG Processing Playground (MetaBCI Demo)")

with st.sidebar:
    st.header("数据源")
    use_dataset = st.checkbox("使用 DREAMER 数据集", value=False)

    if use_dataset:
        subj_id = st.slider("被试编号", 1, 23, 23)
        st.markdown("**标签设置**")
        task_type = st.radio("任务类型", ["二分类", "三分类(VAD)"], horizontal=True)
        label_type = st.selectbox("情绪维度", ["valence", "arousal", "dominance"])
        if task_type == "二分类":
            high_low_split = st.slider("高低阈值", 1.0, 5.0, 3.0, 0.1)
        else:
            st.caption("三分类(VAD)：比较 valence / arousal / dominance 三个分数，最大者作为类别 0/1/2")
            high_low_split = None

    st.header("采集参数")
    fs = st.slider("采样率 (Hz)", 64, 512, 128, 64, disabled=use_dataset)
    duration = st.slider("信号时长 (s)", 1, 10, 3, disabled=use_dataset)

    st.header("预处理")
    l_freq = st.number_input("带通下限 (Hz)", 0.1, 60.0, 1.0)
    h_freq = st.number_input("带通上限 (Hz)", 5.0, 60.0, 45.0)
    do_integrate = st.checkbox("积分", value=False)

    st.header("MLP 结构")
    hidden1 = st.slider("隐藏层1 神经元", 1, 128, 8)
    hidden2 = st.slider("隐藏层2 神经元 (0 = 无)", 0, 128, 0)

    st.header("训练参数")
    epochs = st.slider("训练轮数", 1, 1000, 20)
    lr = st.number_input("学习率", 1e-5, 1.0, 0.01, format="%f")
    dropout_rate = st.slider("Dropout", 0.0, 0.9, 0.2, 0.05)
    test_ratio = st.slider("测试集比例", 0.1, 0.9, 0.2)

    st.header("窗口设置")
    win_sec = st.number_input("窗口长度 (秒)", 0.25, 10.0, 1.0, 0.25)
    hop_sec = st.number_input("步长 (秒)", 0.1, 10.0, 0.5, 0.1)

    st.header("通道选择")
    all_ch = list(range(14))
    sel_channels = st.multiselect("选择用于特征的通道 (索引)", all_ch, default=[0,1,2])

    channel_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
    sel_names = [channel_names[i] for i in sel_channels]

    st.header("特征槽配置")
    # 支持常用脑电频段功率
    feat_types = [
        "Raw",
        "Filtered",
        "Integrated",
        "PSD",
        "Bandpower",  # 宽带功率
        "Bandpower_Theta",
        "Bandpower_Alpha",
        "Bandpower_Beta",
        "Bandpower_Gamma",
        "AlphaDiff",  # AF3-Alpha minus AF4-Alpha
    ]

    max_slots = 12
    num_slots = st.slider("特征槽数量", 1, max_slots, 3)
    slot_cfg = []
    for i in range(num_slots):
        st.markdown(f"**槽 {i+1}**")
        ch = st.selectbox(
            "通道", sel_channels, key=f"slot_ch_{i}", format_func=lambda x: f"{x} ({channel_names[x]})"
        )
        tp = st.selectbox("类型", feat_types, key=f"slot_tp_{i}")
        slot_cfg.append((ch, tp))

    st.header("可视化通道")
    vis_ch = st.selectbox("选择在图中展示的通道", sel_channels, key="vis_ch", format_func=lambda i: f"{i} ({channel_names[i]})")

    run_btn = st.button("运行 Pipeline")

if run_btn:
    if use_dataset:
        dreamer = DREAMER(subjects=[subj_id])
        subj_data = dreamer._get_single_subject_data(subj_id)
        raws = [d["run_1"] for d in subj_data.values()]
        raw_concat = mne.concatenate_raws(raws)
        fs = int(raw_concat.info["sfreq"])
        data_raw = raw_concat.get_data()  # (14, n_times)
        # 根据标签类型获得每段视频标签
        labels_per_vid = []
        for sess_key, raw in subj_data.items():
            raw_obj = raw["run_1"]
            # raw_obj.annotations.description 只保存 valence 分数字符串
            if task_type == "二分类":
                try:
                    score_val = float(raw_obj.annotations.description[0])
                except ValueError:
                    continue
                label_bin = 1 if score_val > high_low_split else 0
                labels_per_vid.append(label_bin)
            else:
                # 读取.mat 以获取三个维度分数
                mat_path = dreamer.data_path(subj_id)
                try:
                    mdata = mat73.loadmat(mat_path)
                except:
                    mdata = sio.loadmat(mat_path)
                sub_struct = (mdata['DREAMER']['Data'] if 'DREAMER' in mdata else mdata['Data'])[subj_id-1]
                # helper
                def _get(arr, key):
                    return arr[key] if isinstance(arr, dict) else getattr(arr, key)
                v_arr = _get(sub_struct, 'ScoreValence').flatten()
                a_arr = _get(sub_struct, 'ScoreArousal').flatten()
                d_arr = _get(sub_struct, 'ScoreDominance').flatten()
                scores_triplet = np.stack([v_arr, a_arr, d_arr], axis=1) # (18,3)
                for trip in scores_triplet:
                    labels_per_vid.append(int(np.argmax(trip)))
                break  # labels_per_vid filled
    else:
        data_raw = generate_dummy_eeg(duration, fs)

    data_proc = bandpass_filter(data_raw, fs, l_freq, h_freq)
    if do_integrate:
        data_proc = optional_integrate(data_proc)

    data_proc_sel = data_proc[np.array(sel_channels)]

    win_len = int(fs * win_sec)
    hop_len = max(1, int(fs * hop_sec))

    def extract_scalar(channel_data: np.ndarray, tp: str):
        if tp == "Raw":
            return channel_data.mean()
        if tp == "Filtered":
            return bandpass_filter(channel_data[np.newaxis,:], fs, l_freq, h_freq).mean()
        if tp == "Integrated":
            return optional_integrate(channel_data[np.newaxis,:]).mean()
        if tp == "PSD":
            f, pxx = signal.welch(channel_data, fs=fs, nperseg=len(channel_data))
            return pxx.mean()
        if tp.startswith("Bandpower"):
            # 频段映射
            bands = {
                "Bandpower": (l_freq, h_freq),
                "Bandpower_Theta": (4, 8),
                "Bandpower_Alpha": (8, 13),
                "Bandpower_Beta": (13, 30),
                "Bandpower_Gamma": (30, 45),
            }
            low, high = bands.get(tp, (l_freq, h_freq))
            # 带通滤波后功率
            bp = bandpass_filter(channel_data[np.newaxis, :], fs, low, high)[0]
            return np.mean(bp ** 2)
        if tp == "AlphaDiff":
            # Ensure AF3 idx 0 and AF4 idx 13 per channel_names list
            idx_af3 = channel_names.index('AF3')
            idx_af4 = channel_names.index('AF4')
            alpha_low, alpha_high = 8, 13
            pow_af3 = bandpass_filter(channel_data[np.newaxis, idx_af3], fs, alpha_low, alpha_high)[0]
            pow_af4 = bandpass_filter(channel_data[np.newaxis, idx_af4], fs, alpha_low, alpha_high)[0]
            return np.mean(pow_af3 ** 2) - np.mean(pow_af4 ** 2)
        return 0.0

    def build_features(segment: np.ndarray):
        # segment shape (n_sel_channels, win_len)
        vec = []
        for ch_idx, tp in slot_cfg:
            vec.append(extract_scalar(segment[sel_channels.index(ch_idx)], tp))
        return np.array(vec)

    # Determine input dimension via first segment
    test_seg = data_proc_sel[:, :win_len]
    in_dim = len(slot_cfg)

    # 初始化模型
    out_dim = 3 if (use_dataset and 'task_type' in locals() and task_type=="三分类(VAD)") else 1
    model = SimpleMLP(in_dim, hidden1, hidden2, out_dim, dropout_rate)

    X = []
    for start in range(0, data_proc_sel.shape[1] - win_len + 1, hop_len):
        seg = data_proc_sel[:, start:start+win_len]
        X.append(build_features(seg))
    X = np.stack(X)  # (samples, channels)

    if use_dataset:
        # 重新遍历窗口，分配标签
        # 先计算每窗口起始位置对应的是第几个视频段
        window_labels = []
        # 获取每个视频段的样本范围
        vid_boundaries = []
        cursor = 0
        for raw in raws:
            n_time = raw.n_times
            vid_boundaries.append((cursor, cursor + n_time))
            cursor += n_time
        # 对每个窗口，根据起始时间找到所属视频
        for start in range(0, data_proc_sel.shape[1] - win_len + 1, hop_len):
            t_idx = start  # 采样点
            # 找所属视频
            vid_idx = next(i for i,(s,e) in enumerate(vid_boundaries) if s<=t_idx<e)
            window_labels.append(labels_per_vid[vid_idx])
        y = np.array(window_labels, dtype=np.int64 if task_type=="三分类(VAD)" else np.float32)
    else:
        # 简单标签: 功率总和高于中位数 -> 1
        power_sum = X.sum(axis=1)
        median = np.median(power_sum)
        y = (power_sum > median).astype(np.float32)

    # ------------------ 数据划分 ------------------
    n_samples = len(X)
    n_classes = len(np.unique(y))
    # 若原比例下测试样本不足 n_classes，则扩充到至少每类一个样本
    test_size = max(n_classes, int(n_samples * test_ratio))
    # 避免 test_size 过大或等于全部样本
    if test_size >= n_samples:
        test_size = n_classes if n_samples > n_classes else 1

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError as e:
        # 当仍无法满足分层要求时退回非分层随机划分
        st.warning(f"分层划分失败: {e}，已改为随机划分")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )

    # --------- 标准化 ---------
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    if out_dim == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        if out_dim == 1:
            loss = criterion(pred, y_train_t)
        else:
            loss = criterion(pred, y_train_t.squeeze().long())
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    # 评估
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        if out_dim == 1:
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).float().numpy()
        else:
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).numpy()
        acc = accuracy_score(y_test, preds)
        bacc = balanced_accuracy_score(y_test, preds)

    # Plot loss curve
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_history)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss')

    # Tabs
    tab_sig, tab_mlp = st.tabs(["信号可视化", "MLP 可视化"])

    with tab_sig:
        st.subheader(f"原始 & 预处理对比 (通道 {vis_ch} - {channel_names[vis_ch]})")
        plot_timeseries(data_raw, fs, "Raw EEG", vis_ch)
        plot_psd(data_raw, fs, "Raw PSD", vis_ch)
        plot_timeseries(data_proc, fs, "Processed EEG", vis_ch)
        plot_psd(data_proc, fs, "Processed PSD", vis_ch)

        # ---- Alpha band可视化 (AF3, AF4 及其差) ----
        st.markdown("### Alpha 波 (8-13 Hz) 可视化：AF3, AF4 及差分")
        alpha_low, alpha_high = 8, 13
        idx_af3 = channel_names.index('AF3')
        idx_af4 = channel_names.index('AF4')
        af3_alpha = bandpass_filter(data_raw[idx_af3:idx_af3+1], fs, alpha_low, alpha_high)
        af4_alpha = bandpass_filter(data_raw[idx_af4:idx_af4+1], fs, alpha_low, alpha_high)
        diff_alpha = np.abs(af3_alpha - af4_alpha)

        plot_timeseries(af3_alpha, fs, "AF3 Alpha (8-13 Hz)")
        plot_timeseries(af4_alpha, fs, "AF4 Alpha (8-13 Hz)")
        plot_timeseries(diff_alpha, fs, "|AlphaDiff| (|AF3 - AF4|)")

    with tab_mlp:
        st.subheader("网络结构示意")
        dot = f"""
        digraph G {{
          rankdir=LR;
          input[label=\"{in_dim} feats\", shape=box, style=filled, fillcolor=lightgray];
          h1[label=\"Hidden1\n{hidden1} neurons\", shape=ellipse, style=filled, fillcolor=lightblue];
          { 'h2[label="Hidden2\n'+str(hidden2)+' neurons", shape=ellipse, style=filled, fillcolor=lightblue];\n          input->h1->h2->out;' if hidden2>0 else 'input->h1->out;' }
        }}"""
        st.graphviz_chart(dot)
        st.subheader("模型输出（示例）")
        with torch.no_grad():
            sample_logits = model(X_test_t[:1])
            if out_dim == 1:
                sample_prob = torch.sigmoid(sample_logits).item()
                st.metric("类别 1 概率", f"{sample_prob:.3f}")
            else:
                sample_prob = torch.softmax(sample_logits, dim=1).numpy()[0]
                st.metric("Softmax V/A/D", f"{sample_prob}")
        st.caption("拖动侧边栏滑块可立即观察结构与输出变化")
        st.pyplot(fig_loss)
        st.metric("测试准确率", f"{acc*100:.2f}%")
        st.metric("Balanced Acc", f"{bacc*100:.2f}%")
        st.markdown(f"**任务说明**：将 1 秒窗口的选定通道功率总和高于中位数判为 **1(高功率)**，否则 **0(低功率)**。")
        st.markdown("输入特征配置:" + ", ".join([f"槽{i+1}:{channel_names[ch]}-{tp}" for i,(ch,tp) in enumerate(slot_cfg)]))
        st.markdown(f"输入维度: {in_dim}")
        st.markdown(f"训练集/测试集样本: {len(y_train)}/{len(y_test)}")

        # 显示权重（仅二分类时）
        if out_dim == 1:
            st.subheader("网络权重（线性层）")
            w1 = model.layers[0].weight.detach().cpu().numpy()
            df_w1 = pd.DataFrame(w1, columns=[f"feat{i}" for i in range(in_dim)], index=[f"h1_{i}" for i in range(hidden1)])
            st.markdown("**隐藏层1 权重**")
            st.dataframe(df_w1)

            if hidden2 > 0:
                w2 = model.layers[3].weight.detach().cpu().numpy()
                df_w2 = pd.DataFrame(w2, columns=[f"h1_{i}" for i in range(hidden1)], index=[f"h2_{i}" for i in range(hidden2)])
                st.markdown("**隐藏层2 权重**")
                st.dataframe(df_w2)

                w_out = model.layers[-1].weight.detach().cpu().numpy()
                df_out = pd.DataFrame(w_out, columns=[f"h2_{i}" for i in range(hidden2)], index=["output"])
                st.markdown("**输出层 权重**")
                st.dataframe(df_out)

            # --- 根据权重映射电阻 ---
            Rs = 10000.0  # 10 kOhm reference sense resistor
            st.markdown(f"### 电阻映射 (参考电阻 Rs = {Rs/1000:.0f} kΩ)")

            def weight_to_resistor(w):
                if w == 0:
                    return np.inf
                return Rs / abs(w)

            # 只对隐藏层1权重做示例映射
            resistors = np.vectorize(weight_to_resistor)(w1)
            signs = np.sign(w1)
            df_R = pd.DataFrame(resistors/1000, columns=df_w1.columns, index=df_w1.index)  # 转kΩ
            df_sign = pd.DataFrame(signs, columns=df_w1.columns, index=df_w1.index)

            st.markdown("**电阻值 (kΩ, 四舍五入一位)**")
            st.dataframe(df_R.round(1))
            st.markdown("**符号矩阵 (1=正, -1=负)**")
            st.dataframe(df_sign)

            # --- 综合贡献度计算 ---
            out_w = model.layers[-1].weight.detach().cpu().numpy().flatten()  # shape (hidden1,)
            contrib = np.abs(out_w[:, None] * w1)  # (hidden1, in_dim)
            feat_import = contrib.sum(axis=0)
            perc = feat_import / (feat_import.sum() + 1e-12)
            df_import = pd.DataFrame({
                'feature': df_w1.columns,
                'abs_weight_sum': np.round(feat_import, 4),
                'ratio_%': np.round(perc * 100, 1)
            })
            st.markdown("**输入特征综合贡献度** (输出层权重 × 隐藏层权重绝对值求和)")
            st.dataframe(df_import)

            # ================= 电路仿真工具 =================
            st.markdown("## 简易电路仿真")
            with st.expander("输入特征电压 (μV) 并预测"):
                with st.form("sim_form"):
                    v0 = st.number_input("feat0 (μV)", value=10.0, key="v0")
                    v1 = st.number_input("feat1 (μV)", value=10.0, key="v1")
                    v2 = st.number_input("feat2 (μV)", value=10.0, key="v2")
                    submitted = st.form_submit_button("计算仿真")
                if submitted:
                    x_vec = torch.tensor([[v0*1e-6, v1*1e-6, v2*1e-6]], dtype=torch.float32)  # 转V
                    with torch.no_grad():
                        h_pre = torch.matmul(x_vec, torch.tensor(w1.T)) + model.layers[0].bias  # (1, hidden1)
                        h_act = torch.relu(h_pre)
                        logit = torch.matmul(h_act, torch.tensor(model.layers[-1].weight.T)) + model.layers[-1].bias
                        prob = torch.sigmoid(logit).item()
                    st.write("隐藏层线性输出 z:", h_pre.numpy())
                    st.write("隐藏层 ReLU a:", h_act.numpy())
                    st.write("最终 logit:", float(logit))
                    st.write("Sigmoid 概率:", prob)
                    st.success("分类结果: 1" if prob>0.5 else "分类结果: 0")

                    # 计算电阻网络等效电流 (忽略bias)
                    currents = []
                    for h_idx in range(hidden1):
                        I = 0.0
                        for f_idx, V in enumerate([v0*1e-6, v1*1e-6, v2*1e-6]):
                            w = w1[h_idx, f_idx]
                            if w == 0:
                                continue
                            R = Rs/abs(w)
                            I += np.sign(w)*V / R  # A
                        currents.append(I)
                    st.write("节点电流 I_hidden (A):", currents)

                    # 电流（按 Rs 映射）
                    I_pre = (h_pre.numpy().flatten()) / Rs  # A
                    I_post = (h_act.numpy().flatten()) / Rs
                    st.write("隐藏层电流 (未过ReLU) A:", I_pre)
                    st.write("隐藏层电流 (ReLU后) A:", I_post)

                    I_out = float(logit) / Rs
                    st.write("输出节点等效电流 I_out (A):", I_out)
                    st.caption("分类规则: I_out > 0 (即 logit > 0) 判为 1，否则 0")

    if in_dim != hidden1:
        st.warning(f"输入特征维度为 {in_dim}，而隐藏层1 神经元为 {hidden1}，两者不一致可能导致解释混淆。") 