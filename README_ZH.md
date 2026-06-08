<div align="center">

# 信号特征智能分析

**基于深度学习的信号调制分类与数字信号处理 (DSP)**

[![Model: CNN--Attention](https://img.shields.io/badge/model-cnn--attention-000000.svg?style=flat-square)](https://github.com/AsaqeLee/Signal)
[![Framework: PyTorch](https://img.shields.io/badge/framework-pytorch-000000.svg?style=flat-square)](https://github.com/AsaqeLee/Signal)
[![Status: High--Integrity](https://img.shields.io/badge/status-high--integrity-000000.svg?style=flat-square)](https://github.com/AsaqeLee/Signal)

[English](./README.md) | 简体中文

</div>

---

## 项目简介

**Signal Intelligence** 是一个利用深度学习进行自动调制分类 (AMC) 的先进框架。它集成了多尺度特征融合与注意力机制，能够在极具挑战性的信道条件下（包括多径衰落和频率偏移）准确识别信号类型。

>[!IMPORTANT]
>本项目采用了 Squeeze-and-Excitation (SE) 模块和深度可分离卷积，在保持极低计算开销的同时实现了高精度识别。

---

## 架构流程

该流水线处理从原始信号输入到最终推理的完整过程。

```mermaid
graph LR
    Input[原始 I/Q 信号] --> Aug[数据增强层]
    Aug --> CNN[多尺度 CNN]
    CNN --> Attn[注意力机制]
    Attn --> Classify[调制分类器]
    Classify --> Output[概率分布输出]
    
    style CNN fill:none,stroke:#000,stroke-width:2px
    style Attn fill:none,stroke:#000,stroke-width:2px
```

---

## 技术规格

<details>
<summary><b>模型内部结构</b></summary>

```text
project/
├── model/
│   ├── modulation_classifier.py  # 核心编排层
│   ├── modules.py                # SE 模块与深度卷积
│   └── base_model.py             # 特征提取骨干网络
├── data/
│   ├── dataset.py                # 高效数据加载器
│   └── augmentations.py          # 相位与频率抖动
└── utils/                        # 评估指标与早停机制
```
</details>

<details>
<summary><b>强约束训练策略</b></summary>

系统采用了多种先进技术以确保模型的收敛性与泛化能力：
- **优化器:** 带有权重衰减的 AdamW。
- **调度器:** 余弦退火 (Cosine Annealing) 学习率策略。
- **正则化:** 标签平滑 (Label Smoothing) 与特征规范化。
- **精度:** 支持 FP16 混合精度训练以加速迭代。
</details>

<details>
<summary><b>安装与训练</b></summary>

### 前置要求
- Python 3.9 或更高版本
- 支持 CUDA 的 GPU (强烈推荐)

### 执行指令
```bash
# 安装依赖
pip install -r requirements.txt

# 启动训练
python -m project.train
```
</details>

---

## 核心能力

- **高鲁棒性 DSP:** 能够有效抵抗多径效应和频率漂移。
- **特征融合:** 结合全局与局部特征，提升分类精度。
- **自适应增强:** 在训练过程中动态注入信号噪声，增强模型健壮性。

---

<div align="center">

&copy; 2026 AsaqeLee. 为先进信号处理研究而设计。

</div>
