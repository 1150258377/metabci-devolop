# MetaBCI

## Welcome! 
MetaBCI is an open-source platform for non-invasive brain computer interface. The project of MetaBCI is led by Prof. Minpeng Xu from Tianjin University, China. MetaBCI has 3 main parts:
* brainda: for importing dataset, pre-processing EEG data and implementing EEG decoding algorithms.
* brainflow: a high speed EEG online data processing framework.
* brainstim: a simple and efficient BCI experiment paradigms design module. 

This is the first release of MetaBCI, our team will continue to maintain the repository. If you need the handbook of this repository, please contact us by sending email to TBC_TJU_2022@163.com with the following information:
* Name of your teamleader
* Name of your university(or organization)

We will send you a copy of the handbook as soon as we receive your information.

## Paper

If you find MetaBCI useful in your research, please cite:

Mei, J., Luo, R., Xu, L., Zhao, W., Wen, S., Wang, K., ... & Ming, D. (2023). MetaBCI: An open-source platform for brain-computer interfaces. Computers in Biology and Medicine, 107806.

And this open access paper can be found here: [MetaBCI](https://www.sciencedirect.com/science/article/pii/S0010482523012714)

## Content

- [MetaBCI](#metabci)
  - [Welcome!](#welcome)
  - [Paper](#paper)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Features](#features)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [What do we need?](#what-do-we-need)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

* BCI datasets come in different formats and standards
* It's tedious to figure out the details of the data
* Lack of python implementations of modern decoding algorithms
* It's not an easy thing to perform BCI experiments especially for the online ones.

If someone new to the BCI wants to do some interesting research, most of their time would be spent on preprocessing the data, reproducing the algorithm in the paper, and also find it difficult to bring the algorithms into BCI experiments.

### The solution

The Meta-BCI will:

* Allow users to load the data easily without knowing the details
* Provide flexible hook functions to control the preprocessing flow
* Provide the latest decoding algorithms
* Provide the experiment UI for different paradigms (e.g. MI, P300 and SSVEP)
* Provide the online data acquiring pipeline.
* Allow users to bring their pre-trained models to the online decoding pipeline.

The goal of the Meta-BCI is to make researchers focus on improving their own BCI algorithms and performing their experiments without wasting too much time on preliminary preparations.

## Features

* Improvements to MOABB APIs
   - add hook functions to control the preprocessing flow more easily
   - use joblib to accelerate the data loading
   - add proxy options for network connection issues
   - add more information in the meta of data
   - other small changes

* Supported Datasets
   - MI Datasets
     - AlexMI
     - BNCI2014001, BNCI2014004
     - PhysionetMI, PhysionetME
     - Cho2017
     - MunichMI
     - Schirrmeister2017
     - Weibo2014
     - Zhou2016
   - SSVEP Datasets
     - Nakanishi2015
     - Wang2016
     - BETA

* Implemented BCI algorithms
   - Decomposition Methods
     - SPoC, CSP, MultiCSP and FBCSP
     - CCA, itCCA, MsCCA, ExtendCCA, ttCCA, MsetCCA, MsetCCA-R, TRCA, TRCA-R, SSCOR and TDCA
     - DSP
   - Manifold Learning
     - Basic Riemannian Geometry operations
     - Alignment methods
     - Riemann Procustes Analysis
   - Deep Learning
     - ShallowConvNet
     - EEGNet
     - ConvCA
     - GuneyNet
     - Cross dataset transfer learning based on pre-training
   - Transfer Learning
     - MEKT
     - LST

## DREAMER Emotion Recognition Demo (New)

This release integrates the DREAMER emotion-recognition dataset into MetaBCI.  
You can now load the raw DREAMER `.mat` file and quickly evaluate a simple EEG-based
emotion-classification pipeline that extracts frequency–band power features and
trains an SVM.

### 1  Prerequisites

```bash
# basic dependencies (already listed in requirements.txt)
pip install mne scikit-learn scipy mat73
```

### 2  Dataset placement

Download `DREAMER.mat` from the official website and copy / move it to the root of
the repository (the same folder that contains `README.md`). The demo searches this
exact path and therefore requires no extra arguments.

### 3  Run the demo

```bash
# from the project root
python MetaBCI-master/demos/dreamer_analysis.py
```

The script will

1. Load EEG for the specified subject (default **1**)
2. Band-pass filter (1–45 Hz)
3. Compute band-power features (θ, α, β, γ) using the new helper
   `metabci.brainda.algorithms.feature_analysis.bandpower`
4. Train an RBF-kernel SVM and print an accuracy / classification report.

Feel free to edit `subject_id`, change frequency bands or swap the classifier to
explore the dataset.

### 4  Code changes behind the scenes

* **metabci/brainda/datasets/dreamer.py** – new robust loader supporting both
  `scipy.io` & `mat73` structures.
* **datasets/__init__.py** – cleaned invalid imports that previously caused
  `ImportError`.
* **algorithms/feature_analysis/freq_analysis.py** – added `bandpower` and
  re-exported it via `feature_analysis.__init__`.
* **demos/dreamer_analysis.py** – end-to-end example described above.

These edits have no impact on existing MI / P300 / SSVEP pipelines.

### 5  How to run & tweak

* **Run out-of-the-box**
  ```bash
  python MetaBCI-master/demos/dreamer_analysis.py
  ```
  The script loads data, extracts band-power features and trains an RBF-SVM.

* **Change the classifier**  – open the same file and locate
  ```python
  clf = SVC(kernel='rbf', random_state=42)
  clf.fit(X_train, y_train)
  ```
  Replace the estimator with any classifier from `scikit-learn`, or plug in your
  own deep-learning model.

* **Labels** – by default we do binary Valence classification (score > 3 → 1,
  else 0). Edit the `label` line (~44) to create multi-class labels or switch to
  Arousal / Dominance.

* **Pipeline recap**
  1. Load one subject via `DREAMER` loader.
  2. 1–45 Hz band-pass filter.
  3. Compute θ/α/β/γ band power for every channel.
  4. Train classifier & evaluate.

---

Enjoy exploring emotion-related EEG with MetaBCI 🎉

## Interactive EEG Playground (Streamlit)

`demos/eeg_platform_streamlit.py` 提供了一个「**傻瓜式**」的可视化实验平台，可一键切换 **随机模拟信号** 与 **DREAMER 真实数据**，并在浏览器中即时查看波形、PSD、特征与 MLP 训练过程。

### 1  真实数据读取

1. 勾选侧边栏 `使用 DREAMER 数据集` 选项。
2. 选择被试编号 *(1–23)*、标签维度 *(valence / arousal / dominance)* 以及高/低阈值 *(默认 3.0)*。
3. 代码片段
   ```python
   dreamer = DREAMER(subjects=[subj_id])
   subj_data = dreamer._get_single_subject_data(subj_id)
   raws = [d["run_1"] for d in subj_data.values()]   # 18 视频段
   raw_concat = mne.concatenate_raws(raws)            # -> (n_ch, n_time)
   data_raw = raw_concat.get_data()
   labels_per_vid = [1 if float(r.annotations.description[0])>thr else 0
                     for r in raws]
   ```
   *原始结构*：23 名被试 × 18 段视频，每段附带 3 维连续评分 (1–5)。
   *当前简化*：对选定维度二值化 (score>阈值 → 1，高情绪)。

### 2  预处理 → 特征 → 滑窗 → 标签对齐

1. **带通滤波**：用户自定义 
   ```python
   data_proc = bandpass_filter(data_raw, fs, l_freq, h_freq)
   ```
2. **(可选) 积分**：`np.cumsum` 作积分特征。  
3. **滑动窗口**：
   ```python
   win_len = int(fs * win_sec)    # 窗口长度
   hop_len = int(fs * hop_sec)    # 步长
   for start in range(0, n_time-win_len+1, hop_len):
       seg = data_proc_sel[:, start:start+win_len]
   ```
4. **标签映射**：窗口起点定位到所属视频段
   ```python
   vid_idx = next(i for i,(s,e) in boundaries if s<=start<e)
   y.append(labels_per_vid[vid_idx])
   ```
   这样即使 **时间被重新切片**，仍保持 *window ↔ 原视频标签* 的 1-to-1 对应。

### 3  特征槽 (可自定义 1–12 维)

- 类型：`Raw`, `Filtered`, `Integrated`, `PSD`, `Bandpower`, 以及 θ / α / β / γ 四档功率。
- 每槽 = (通道, 特征类型)。界面拖动即可添加/删除。
- `in_dim = len(slot_cfg)` 自动适配网络输入。

### 4  MLP 训练 & 评估

- 二层 MLP，隐藏层大小可调，训练轮数上限 1000。
- 自动分层 `train_test_split`（不足时降级到随机划分）。
- 实时绘制 Loss 曲线 + Test Accuracy + 单样本概率。

### 5  代码美观 & 逻辑顺序

```
┌ Sidebar (参数与特征配置)
│
├── 数据源选择 (随机 / DREAMER)
├── 采样 & 预处理参数
├── 特征槽 (动态数量)
└── 训练参数

┌ Tabs
│
├── 信号可视化 (Raw / Processed / PSD)
└── MLP 可视化 (Graphviz 结构 + 训练曲线 + Metric)
```
- 模块化函数 (`bandpass_filter`, `extract_scalar`, `build_features`) 保持主流程清晰。
- 所有可调参数实时生效，无需重启。
- 关键异常（如分层错误）自动降级并在界面提示 `st.warning`。

> **小结**：脚本完整复用了 <u>真实情绪标签</u>，在滑窗后仍精确映射；用户可自由增删特征槽、调整窗口与网络结构，即刻看到对分类性能的影响。

---

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TBC-TJU/MetaBCI.git
   ```
2. Change to the project directory
   ```sh
   cd MetaBCI
   ```
3. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
4. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```
## Who are we?

The MetaBCI project is carried out by researchers from 
- Academy of Medical Engineering and Translational Medicine, Tianjin University, China
- Tianjin Brain Center, China


## What do we need?

**You**! In whatever way you can help.

We need expertise in programming, user experience, software sustainability, documentation and technical writing and project management.

We'd love your feedback along the way.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU General Public License v2.0 License. See `LICENSE` for more information.

## Contact

Email: TBC_TJU_2022@163.com

## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
