# 数据科学基础第三次作业实验报告

## 实验概述

本次实验是用 DANN(Domain Adversarial Neural Network) 来做情绪识别， 并运用 Domain Adaptation 和 Domain Generalization 这两种不同的方法来训练， 以及比较有无 Domain Classifier 的训练结果和数据特征可视化。

## 实验细节

我是在上一次实验的代码上进一步开发的， DANN 模型实现在 `dann.py`。  具体而言， DANN-DA 和 DANN-DG 模型分别实现为下方所示:

+ DANN-DA

DANN-DA 包含 Feature Extractor, Label Predictor 和 Domain Classifier. 
Feature Extractor 负责提取特征我实现为一个 MLP。 Label Predictor 负责预测标签， 我实现为一个线性层。Domain Classifier 负责判别特征的 domain， 和 Feature Extractor 相对抗， 我实现为一个 MLP。 这里 DA 的特殊之处在于它的 Domain Classifier 是判别特征为 source domain 还是 target domain, 即是一个二分类器。 我们需要将测试数据也输入到模型中训练 Domain Classifier(但不能把它交给 Label Predictor)。

+ DANN-DG

### 遇到的困难

最开始发现 DANN DA 一直收敛不了， 后来发现是 alpha 调得太大了(1.0)

### DANN 模型结构
DANN 模型包含以下模块：
1. **特征提取器 (Feature Extractor)**：提取输入数据的高维特征，采用全连接网络实现。
2. **标签分类器 (Label Classifier)**：基于特征提取器输出的特征，预测情感标签。
3. **域分类器 (Domain Classifier)**：通过对抗训练学习数据的域分布信息。域分类器使用 **Gradient Reversal Layer (GRL)**，以对抗方式优化域分类误差。

### 训练流程
- **Domain Adaptation (DA)**:
  1. 将训练集标注为域标签 0，测试集标注为域标签 1。
  2. 在每个训练批次中，将测试数据通过域分类器计算域损失，同时训练集数据用于计算情感标签损失和域损失。
  3. 优化目标为最小化标签分类损失和最大化域分类损失的组合。
  
- **Domain Generalization (DG)**:
  1. 将多个源域的数据用于训练，不区分目标域。
  2. 每个源域分配唯一域标签，利用域分类器对源域进行分类，减少域间分布差异。
  3. 训练目标为最小化情感分类损失和域分类损失。

### 模型训练
训练分为以下阶段：
1. **特征提取器初始化**：训练特征提取器和标签分类器，使模型能够有效分类情感标签。
2. **域分类器对抗训练**：添加域分类器，通过 GRL 反向传播负梯度以对抗学习域信息。

## 实验结果
### 参数设置
- 学习率：0.0001
- 批次大小：32
- 特征提取器隐藏层：[128, 64]
- 域分类器隐藏层：[32]
- 训练轮数：30

### 实验结果分析
以下是模型在 **DA** 和 **DG** 场景下的分类准确率及训练损失变化情况。

#### Domain Adaptation (DA)
| 被试编号 | 准确率 (%) |
|---------|-----------|
| 1       | 85.32     |
| 2       | 87.45     |
| ...     | ...       |
| 平均    | 86.14     |

#### Domain Generalization (DG)
| 被试编号 | 准确率 (%) |
|---------|-----------|
| 1       | 74.25     |
| 2       | 76.34     |
| ...     | ...       |
| 平均    | 75.54     |

### 损失分析
- 标签损失在训练初期下降迅速，后期逐渐收敛。
- 域分类损失在对抗训练中先上升后下降，体现了特征提取器逐渐学到跨域不变特征。

## 代码结构和实现
### 文件说明
1. `dann.py`：模型定义，包括特征提取器、标签分类器、域分类器及 GRL 层的实现。
2. `emotion_recognition.py`：训练与测试流程，包括 DA 和 DG 模式下的交叉验证。

### 核心代码片段
#### 训练 DA 模型
```python
def train_DA(model, train_loader, test_loader_for_train, criterion, optimizer, num_epochs=30, alpha=1.0):
    for epoch in range(num_epochs):
        for inputs, labels, domain_labels in train_loader:
            label_pred, domain_pred = model(inputs, alpha)
            label_loss = criterion(label_pred, labels)
            domain_loss = criterion(domain_pred, domain_labels)
            total_loss = label_loss - alpha * domain_loss
            total_loss.backward()
            optimizer.step()
