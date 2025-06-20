# MetaBCI 中文简介

## 项目介绍
MetaBCI 是一个面向非侵入式脑机接口 (BCI) 研究的开源平台，由天津大学徐敏鹏教授团队主导开发。本仓库包含三大模块：

* **brainda**：数据集加载、预处理与各类 EEG 解码算法
* **brainflow**：高速 EEG 在线数据处理框架
* **brainstim**：简洁高效的 BCI 实验范式设计模块

---

## 新增功能 —— DREAMER 情绪识别示例
本次更新已将 **DREAMER** 情绪识别数据集整合进 MetaBCI，可直接运行示例脚本完成特征提取与情绪分类。

### 1. 环境依赖
在已有 `requirements.txt` 基础上，如缺失请安装：

```bash
pip install mne scikit-learn scipy mat73
```

### 2. 数据集放置
将官方提供的 `DREAMER.mat` 下载后**放置到仓库根目录**（与 `README.md` 同级）。脚本默认从该路径读取，无需额外参数。

### 3. 运行示例

```bash
# 于项目根目录执行
python MetaBCI-master/demos/dreamer_analysis.py
```
脚本流程：
1. 加载指定被试的 EEG 数据（默认被试 **1**）
2. 1–45 Hz 带通滤波
3. 使用新增的 `bandpower` 函数计算 θ/α/β/γ 频段能量
4. 训练 RBF-kernel SVM，输出准确率与分类报告

你可以修改 `demos/dreamer_analysis.py` 中的 `subject_id`、频段定义或替换分类器，以探索不同配置。

### 4. 关键代码变更
| 文件 | 说明 |
|------|------|
| `metabci/brainda/datasets/dreamer.py` | 新增健壮的 DREAMER 加载器，兼容 `scipy.io` 与 `mat73` 解析结构 |
| `metabci/brainda/datasets/__init__.py` | 移除失效导入，避免 `ImportError` |
| `metabci/brainda/algorithms/feature_analysis/freq_analysis.py` | 增加 `bandpower` 函数，并在 `__init__` 中导出 |
| `demos/dreamer_analysis.py` | 全流程示例脚本 |

---

## 使用说明

### 一键运行
- **加载数据并完成分类**：
  ```bash
  python MetaBCI-master/demos/dreamer_analysis.py
  ```
  该脚本会自动完成数据读取、预处理、特征提取与情绪分类。

### 修改/替换分类算法
- 打开 `MetaBCI-master/demos/dreamer_analysis.py`，定位到 **第 67 行左右**：
  ```python
  clf = SVC(kernel='rbf', random_state=42)
  clf.fit(X_train, y_train)
  ```
  - 将 `SVC(...)` 替换为任意 `sklearn` 分类器，或是你自定义的模型。
  - 如果需要深度学习模型，可在此处写入 PyTorch / TensorFlow 代码，只要输入特征 `X_train`、`y_train`，即可与当前流程无缝衔接。

### 标签的来源与处理逻辑
- DREAMER.mat 中包含每个视频的 **情绪自评分** (Valence / Arousal / Dominance)，范围 1~5。
- 示例脚本仅使用 **Valence** 维度进行二分类：
  - `valence > 3` 记为 **1（高）**
  - `valence ≤ 3` 记为 **0（低）**
- 你可以在脚本第 44 行附近修改 `label` 生成逻辑，以实现：
  - 多分类（例如低 / 中 / 高）
  - 使用 Arousal、Dominance 或混合标签

### 整体流程回顾
1. **数据加载**：`metabci.brainda.datasets.DREAMER` 解析 `.mat` 文件，返回 `mne.Raw` 对象。
2. **预处理**：对每个 `Raw` 做 1–45 Hz 带通滤波。
3. **特征提取**：调用 `bandpower` 计算 θ/α/β/γ 四个频段能量；对所有通道展开为特征向量。
4. **训练 / 预测**：默认使用 RBF-SVM；可替换。
5. **评估**：输出准确率与 `classification_report`。

> ⚠️ 如需批量处理全部 23 名被试，可在脚本顶部将 `subject_id` 替换为循环或列表；或直接使用 `DREAMER().get_data()` 批量返回。

---

## 快速安装
```bash
git clone https://github.com/TBC-TJU/MetaBCI.git
cd MetaBCI
pip install -r requirements.txt
pip install -e .
```

---

祝使用愉快，期待你的反馈与贡献！ 