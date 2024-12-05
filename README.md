# 调制分类项目

这是一个使用深度学习进行信号调制分类的项目。

## 项目结构

```
project/
├── config.py          # 配置文件
├── train.py          # 训练脚本
├── data/             # 数据处理相关
│   ├── dataset.py    # 数据集类
│   └── augmentations.py  # 数据增强
├── model/            # 模型相关
│   ├── base_model.py     # 基础模型类
│   ├── modules.py        # 模型组件
│   └── modulation_classifier.py  # 调制分类器
└── utils/            # 工具函数
    ├── metrics.py        # 评估指标
    └── early_stopping.py # 早停机制
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将数据放在`train_data_true`目录下，按照调制类型分类:

```
train_data_true/
├── BPSK/
├── QPSK/
├── 8PSK/
└── ...
```

## 训练模型

```bash
python -m project.train
```

## 主要特性

- 改进的模型架构
  - Squeeze-and-Excitation模块
  - 深度可分离卷积
  - 多尺度特征融合
  - 注意力机制

- 增强的训练策略
  - AdamW优化器
  - 余弦退火学习率
  - 梯度累积
  - 混合精度训练

- 丰富的数据增强
  - 信号噪声
  - 频率偏移
  - 相位噪声
  - 多径效应
  - 频谱增强
  - 随机擦除

## 配置说明

主要配置参数在`config.py`中:

- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `MAX_EPOCHS`: 最大训练轮数
- `FEATURE_DIM`: 特征维度
- `DROPOUT_RATE`: Dropout比率
- 等等

## 模型说明

- 使用改进的卷积神经网络进行特征提取
- 多尺度特征融合提高特征表达能力
- 注意力机制关注重要特征
- 辅助分类器帮助训练
- 标签平滑和特征正则化防止过拟合
