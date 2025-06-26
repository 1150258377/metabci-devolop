# metabci-devolop: 基于 MetaBCI 的交互式脑机接口可视化平台

这是一个基于强大的 **MetaBCI** 框架开发的BCI（脑机接口）应用项目，其核心目标是为脑电数据分析提供一个**交互式的可视化前端**，并集成一个针对 **DREAMER 情感数据集** 的端到端分析流程。

## 核心特性：与 MetaBCI 的区别与本项目的优化

虽然本项目基于 MetaBCI，但它的定位和侧重点有显著不同。如果说 **MetaBCI** 是一个功能全面、底层的BCI算法研究“**引擎**”和“**工具箱**”，那么 **metabci-devolop** 就是一辆基于这个引擎打造的、专注于特定应用场景的“**概念跑车**”。

本项目主要进行了以下优化和功能集成：

### 1. 交互式可视化界面
- **技术实现**：通过集成 **Streamlit** 框架，将复杂的BCI数据分析过程封装成一个简单、直观的Web页面。
- **用户体验**：普通用户不再需要深入研究代码细节，可以直接在浏览器页面上进行操作，查看数据波形、特征分布以及分类结果的可视化图表，极大地降低了BCI技术的入门和使用门槛。

### 2. DREAMER 情感数据集“开箱即用”
- **简化流程**：与原始框架需要编写特定脚本来加载数据集不同，本项目实现了对本地 `DREAMER.mat` 文件的直接读取和预处理。
- **易用性**：您只需将 `DREAMER.mat` 数据集文件放置在项目指定目录，即可一键运行完整的情感特征提取和分类流程。

### 3. 端到端的分类与分析演示
- **完整流程**：项目内包含了一个从数据加载、信号预处理、特征提取到执行一个基础分类模型（如SVM）的完整示例。
- **应用场景**：完整演示了如何利用脑电信号对**Valence（愉悦度）**和**Arousal（唤醒度）**等情感状态进行二分类判别，并将分类结果直观地呈现出来。

## 快速开始

1.  **克隆仓库**
    ```bash
    git clone git@github.com:1150258377/metabci-devolop.git
    cd metabci-devolop
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

3.  **准备数据集**
    请将 `DREAMER.mat` 数据文件下载并放置在项目的根目录下。

4.  **运行平台**
    ```bash
    streamlit run MetaBCI-master/demos/eeg_platform_streamlit.py
    ```
    运行后，程序会自动在浏览器中打开一个 Web 页面，您可以在此页面上进行交互式分析。

5.  **使用更复杂的模型**
    ```bash
    python MetaBCI-master/demos/dreamer_big_model.py --subject 1
    ```
    该脚本示范如何替换默认的 SVM，使用 PyTorch 神经网络或自定义模型。若要加载自己
    的微调模型，可通过 `--model path/to/model.pt` 指定模型文件。

6.  **微调并接入大型模型**
    1. 执行 `dreamer_big_model.py` 生成训练特征或直接在脚本中加载 `DREAMER.mat`。
    2. 根据您选择的框架（如 PyTorch、TensorFlow 以及更高级的 **O3** 或 **QWQ** 模型接口）编写训练代码，对特征进行微调。
    3. 微调完成后将模型保存为 `.pt`/`.ckpt` 等文件，并在运行脚本时通过 `--model` 指定路径进行评估。
    4. 若模型部署在远程服务器，可通过 `--remote-url` 和 `--api-key` 发送特征到 HTTP 接口获取预测结果。

    示例：
    ```bash
    # 训练并保存模型权重
    python MetaBCI-master/demos/dreamer_big_model.py --subject 1 --epochs 50 \
        --save-path my_model.pt
    # 加载已有的大模型进行预测
    python MetaBCI-master/demos/dreamer_big_model.py --subject 1 \
        --model my_model.pt
    # 调用远程模型接口进行预测
    python MetaBCI-master/demos/dreamer_big_model.py --subject 1 \
        --remote-url https://api.example.com/predict --api-key YOUR_TOKEN
    ```

### Web 页面功能简介
运行 Streamlit 页面后，你可以在浏览器中完成以下操作：
* **数据源选择**：使用示例信号或加载 DREAMER 数据集；
* **预处理与特征提取**：在侧边栏配置滤波、窗口长度及功率特征；
* **神经网络结构**：动态调整隐藏层数量、Dropout 等超参数；
* **结果可视化与仿真**：查看训练曲线、特征权重，并通过电路仿真输入特征电压获得预测
  结果。

## 项目目标

本项目旨在探索如何将专业的BCI算法框架（MetaBCI）与现代化的Web应用工具（Streamlit）相结合，构建一个用户友好、易于理解和操作的BCI应用原型。它不仅是一个开发工具，更是一个能够让更多人直观感受脑机接口技术魅力的教育和展示平台。
