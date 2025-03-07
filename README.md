# GeoLife轨迹预测系统

本项目缘起 **《数据处理原理与技术》 课程论文**，本项目基于微软GeoLife数据集，实现了一个完整的轨迹预测系统，包括数据预处理、特征工程、模型训练和评估。利用深度学习技术（Pytorch)，系统能够基于历史轨迹点预测用户未来的移动路径。

## 项目特点

- **多阶段数据处理**：原始GPS轨迹数据清洗、分段、异常检测和插值
- **丰富的特征工程**：时间、空间和序列特征提取与选择
- **多种预测模型**：实现并对比了LSTM、Seq2Seq、Transformer和STGCN四种模型
- **灵活的评估体系**：多种评估指标和可视化工具
- **GPU加速支持**：关键计算支持GPU加速，提高处理效率 (建议使用Linux环境以获得完整的RAPIDS库支持)

## 系统架构

```
CopyGeoLife轨迹预测系统/
├── data/                             # 数据目录
│   ├── raw/                          # 原始GeoLife数据
│   └── processed/                    # 处理后的数据
├── preprocess/                       # 预处理模块
│   ├── parse_geolife_plt.py          # PLT文件解析
│   ├── data_cleaning.py              # 数据清洗
│   ├── feature_engineering.py        # 特征工程
│   └── visualization.py              # 数据可视化
├── prediction/                       # 预测模块
│   ├── data/                         # 数据处理
│   │   ├── dataset.py                # 轨迹数据集
│   │   └── dataloader.py             # 数据加载器
│   ├── models/                       # 预测模型
│   │   ├── base_model.py             # 模型基类
│   │   ├── lstm.py                   # LSTM模型
│   │   ├── seq2seq.py                # Seq2Seq模型
│   │   ├── transformer.py            # Transformer模型
│   │   └── stgcn.py                  # 时空图卷积网络
│   ├── utils/                        # 工具函数
│   │   ├── metrics.py                # 评估指标
│   │   ├── losses.py                 # 损失函数
│   │   └── visualization.py          # 预测/消融可视化
│   ├── config.py                     # 配置文件
│   ├── train.py                      # 训练脚本
│   └── eval.py                       # 评估脚本
├── reports/                          # 报告和图表
│   └── figures/                      # 可视化图表
└── outputs/                          # 模型输出
|   ├── models/                       # 保存的模型
|    └── predictions/                  # 预测结果
|—— myenv.yml                          # conda环境迁移
```

## 安装与依赖

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.3+ (可选，用于GPU加速)

### 依赖安装

```
bash
 conda env create -f myenv.yml
```

主要依赖包括：

- pytorch
- numpy
- pandas
- matplotlib
- folium
- scikit-learn
- tqdm

### GPU加速（可选）

若要启用GPU加速，请安装以下包：

```
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install cupy-cuda113
```

## 使用指南

### 数据预处理

1. 解析原始PLT文件：

```
bash

python -m preprocess.parse_geolife_plt
```

2. 数据清洗与分段：

```
bash

python -m preprocess.data_cleaning
```

3. 特征工程与提取：

```
bash
python -m preprocess.feature_engineering
```

### 模型训练

使用以下命令训练不同模型：

```
# LSTM模型
python -m prediction.train --model lstm --input_length 20 --output_length 5

# Seq2Seq模型
python -m prediction.train --model seq2seq --input_length 20 --output_length 5

# Transformer模型
python -m prediction.train --model transformer --input_length 20 --output_length 5

# STGCN模型
python -m prediction.train --model stgcn --input_length 20 --output_length 5
```

可选参数：

- `--config`: 指定配置文件路径
- `--batch_size`: 设置批大小
- `--lr`: 设置学习率
- `--epochs`: 设置训练轮数
- `--patience`: 设置早停耐心值

### 模型评估

单个模型评估：

```
bash
python -m prediction.eval --model path/to/model.pt --mode single
```

模型比较：

```
bash
python -m prediction.eval --model directory/with/models --mode compare
```

消融实验：

```
bash
python -m prediction.eval --model path/to/base_model.pt --mode ablation --ablation_config path/to/config.json
```

## 实验结果与模型权重

- 会存放在output里

## 示例应用

系统可用于多种实际应用场景：

- 交通路径规划和拥堵预测
- 车辆调度和位置预测
- 位置推荐和导航服务
- 移动用户行为分析

## 贡献与开发

欢迎贡献代码或报告问题。开发时请遵循以下准则：

- 遵循PEP 8编码风格
- 提供单元测试
- 文档完善且代码注释充分

## 许可证

本项目采用MIT许可证。

## 致谢

- 感谢数据分析原理与技术课程老师的辛勤教导
- 感谢微软研究院提供的GeoLife数据集
- 感谢开源社区提供的工具和库

## 引用

如果您在研究中使用了本项目，请按如下格式引用：

```
Copy@misc{GeoLife-Trajectory-Prediction,
  author = {Your Name},
  title = {GeoLife轨迹预测系统},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Tsingloong611/GeoLife-Trajectory-Prediction}
}
```
