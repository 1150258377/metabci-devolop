# MAT文件读取器

这是一个用于读取和展示MATLAB .mat文件内容的Python脚本。

## 功能特点

- 读取.mat文件并显示所有变量
- 显示变量的类型、形状和数据类型
- 对于数值数组，显示统计信息（最小值、最大值、平均值、标准差）
- 交互式界面，可以查看特定变量的详细信息
- 支持保存数据到新的.mat文件

## 安装依赖

在运行脚本之前，请先安装所需的Python包：

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install scipy numpy
```

## 使用方法

### 方法1：直接运行脚本
```bash
python read_mat_file.py
```
然后按提示输入.mat文件的路径。

### 方法2：命令行指定文件路径
```bash
python read_mat_file.py "path/to/your/file.mat"
```

### 方法3：在Python中导入使用
```python
from read_mat_file import read_mat_file

# 读取.mat文件
data = read_mat_file("path/to/your/file.mat")

# 访问特定变量
if data:
    variable_name = list(data.keys())[0]  # 获取第一个变量名
    print(f"变量 '{variable_name}' 的内容:")
    print(data[variable_name])
```

## 输出示例

脚本会显示类似以下的信息：

```
正在读取文件: example.mat

==================================================
MAT文件内容分析
==================================================

文件中的变量数量: 3

变量列表:
1. data
   类型: ndarray
   形状: (100, 10)
   数据类型: float64
   最小值: -2.5
   最大值: 2.8
   平均值: 0.1234
   标准差: 1.2345
   前5个元素: [0.1 0.2 0.3 0.4 0.5]
   后5个元素: [2.1 2.2 2.3 2.4 2.5]

2. labels
   类型: ndarray
   形状: (100,)
   数据类型: object
   内容: ['label1' 'label2' 'label3' ...]

==================================================
文件读取完成！
==================================================

选项:
1. 查看特定变量的详细信息
2. 保存变量到新的.mat文件
3. 退出

请选择操作 (1-3):
```

## 注意事项

- 确保您有读取.mat文件的权限
- 对于大型.mat文件，读取可能需要一些时间
- 脚本会自动跳过MATLAB系统变量（以__开头的变量）
- 支持各种数据类型，包括数值数组、字符串数组、结构体等

## 系统要求

- Python 3.6 或更高版本
- scipy 1.7.0 或更高版本
- numpy 1.20.0 或更高版本 

# DREAMER数据集与MetaBCI集成说明

## 1. 项目背景与目标
本项目旨在将公开的DREAMER.mat情感EEG数据集集成到MetaBCI框架中，利用MetaBCI的特征提取和数据处理功能进行分析。通过本集成，用户可以直接用MetaBCI的范式和特征提取工具处理DREAMER数据，实现情感识别等相关研究。

## 2. 数据集结构与解析
- **DREAMER.mat** 包含23名被试，每人观看18个视频片段。
- 每个片段包含EEG数据（采样率128Hz），分为baseline和stimuli两部分。
- 每个片段有情感评分（Valence, Arousal, Dominance）。
- 相关解析脚本：`mat_to_dataframe.py`、`read_mat_file.py`、`debug_dreamer.py`。

## 3. MetaBCI集成方案
### 3.1 新增DREAMER数据集类
- 在`metabci/brainda/datasets/`下实现`dreamer.py`（或在合适位置），继承自MetaBCI的`BaseDataset`。
- 支持自动读取DREAMER.mat，提取每个被试的EEG和情感评分。
- 支持MetaBCI的特征提取（如CSP、FBCSP、频域、时频等）。
- 支持MetaBCI的范式（如EmotionParadigm）。

### 3.2 主要实现细节
- 正确解析mat文件结构，组织为MetaBCI可用的数据格式。
- 提供数据分割、标签提取、通道信息等接口。
- 兼容MetaBCI的pipeline和特征提取流程。

## 4. 测试与演示脚本
- `test_dreamer.py`：测试DREAMER数据加载和预处理。
- `dreamer_analysis.py`：完整分析流程，包括特征提取和SVM分类。
- 运行方法：
  ```bash
  python test_dreamer.py
  python dreamer_analysis.py
  ```

## 5. 依赖安装与环境问题排查
### 5.1 最小化依赖安装
- 推荐只安装核心依赖包，避免不必要的包冲突。
- 主要依赖：`numpy`, `scipy`, `mat73`（如需读取v7.3 mat文件）, `scikit-learn` 等。
- 安装示例：
  ```bash
  pip install numpy scipy scikit-learn mat73
  ```

### 5.2 C++编译环境缺失
- 某些包（如`scikit-learn`）在Windows下需C++编译环境。
- 解决方法：安装[Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)。

### 5.3 mat73缺失
- 如遇`mat73`相关报错，使用`pip install mat73`安装。

### 5.4 包路径不对
- 确认`PYTHONPATH`包含MetaBCI主目录，或在脚本开头添加：
  ```python
  import sys
  sys.path.append('..')  # 或MetaBCI主目录路径
  ```

## 6. 常见问题与排查
- **ImportError/ModuleNotFoundError**：检查依赖是否安装、路径是否正确。
- **mat文件读取失败**：确认mat文件版本，v7.3需用mat73。
- **特征提取报错**：检查数据格式、通道数、标签等是否正确。

## 7. 参考脚本说明
- `mat_to_dataframe.py`：将DREAMER.mat转为pandas DataFrame，便于调试和分析。
- `debug_dreamer.py`：调试DREAMER数据结构和内容。
- `dreamer_metabci_integration.py`：集成MetaBCI流程的主脚本。

---

## AI助手自我介绍
我是基于o3模型的AI助手，在Cursor IDE中为您提供支持。我能够在Cursor IDE中为您提供全方位的支持。不论是编程疑难解答、代码优化建议、技术知识讲解，还是日常开发中的各种任务，我都可以为您提供高效、专业的帮助。无论您遇到什么问题，都可以随时向我提问，我会尽力为您提供最优的解决方案，助力您的开发之路更加顺畅！ 