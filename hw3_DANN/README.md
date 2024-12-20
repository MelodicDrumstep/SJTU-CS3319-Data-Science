# 数据科学基础第三次作业实验报告

## 实验概述

本次实验是用 DANN(Domain Adversarial Neural Network) 来做情绪识别， 并运用 Domain Adaptation 和 Domain Generalization 这两种不同的迁移学习方法来训练， 以及比较有无 Domain Classifier 的训练结果和数据特征可视化。 在下方，我将使用 DA 作为 Domain Adaptation 的简称， DG 作为 Domain Generalization 的简称。

## 实验细节

### 数据预处理

这部分代码请见 [src/data_loader.py](src/data_loader.py) 和 [src/emotion_recognition.py](src/emotion_recognition.py)。代码框架沿用了第二次实验的代码。

与第二次实验的不同之处是，我需要在训练前给数据贴上域的标签。 如果是 DA, 那就给训练集贴上 domain label 为 0， 测试集贴上 domain label 为 1。 如果是 DG， 那就给不同的训练集的被试贴 0~10 的不同标签。

### 模型结构设计

这部分代码请见 [src/dann.py](src/dann.py)

+ DANN-DA

DANN-DA 包含 Feature Extractor, Label Predictor 和 Domain Classifier. 
Feature Extractor 负责提取特征, 我实现为一个 MLP。 Label Predictor 负责预测标签， 我实现为一个线性层。Domain Classifier 负责判别特征的 domain， 和 Feature Extractor 相对抗， 我实现为一个 MLP。 这里 DA 的特殊之处在于它的 Domain Classifier 是判别特征为 source domain 还是 target domain, 即是一个二分类器。 我们需要将测试数据也输入到模型中训练 Domain Classifier(但不能把它交给 Label Predictor)。

+ DANN-DG

DANN-DG 和 DA 的结构类似， 包含 Feature Extractor, Label Predictor 和 Domain Classifier. 
这里 DG 的 Domain Classifier 是判别特征为 source domains 中的哪个 domain, 即是一个 11 分类器。这里我们就不需要提供测试集数据给模型来训练了。

### 训练及测试设置

- Label Predictor 和 Domain Classifier 均使用 CrossEntropyLoss 作为损失函数
- 使用 Adam 作为选择的优化器，应用 0.0001 的学习率
- 每个训练进行 30 个 epoch

### 超参设置

经过小范围的网格搜索(算力不是很充足)， 我选取了以下超参数:

+ num_epochs = 30

+ alpha = 0.1

+ DANN-DA 的 feature extractor 神经元维度为 `[64, 32]`, domain classifier 神经元维度为 `[8]`

+ DANN-DG 的 feature extractor 神经元维度为 `[128, 64]`, domain classifier 神经元维度为 `[32]`

## 实验结果

| 模型 | 平均准确率 | 标准差 |
|-----------|---------------|---------|
| DANN without domain classifier       | 0.48 | 0.12 |
| DANN-DA      | 0.51 | 0.12 |
| DANN-DG       | 0.60 | 0.13 |

## 实验结果分析

我训练出来的 DANN-DG / DANN-DA 在启用 domain classifier 的情况下识别准确率都有一定的提升，DANN-DG 的提升更显著， 但整体识别准确率都不高。 

对于整体识别准确率都不高的原因， 一方面是因为数据集可能并不易于训练， 且跨被试留一交叉验证的条件太苛刻。 此外， 可能我的代码实现中有不恰当但没有被我发现的地方， 超参设置可能也不合理。

对于 DANN-DG 与 DANN-DA 的识别准确率提升的问题，是 Domain Classifier 增强了模型的泛化能力。而 DANN-DG 的训练效果更好， 我认为是这个任务比较适合用分类 source domains 的 domain classifier 来和 feature extractor 对抗训练。

## 数据可视化及分析

以上数据特征均来自跨被试交叉逐一验证的最后一折。

+ DANN without domain classifier

<img src="images/DANN%20without%20domain%20classifier.png" alt="DANNW" width="50%">

可以看出 class 2 分类得比较好, 其他两个 class 没怎么分开， 整体特征分布看起来较为混乱，分类边界模糊。
可能是因为没有使用领域分类器时，模型无法有效地对不同领域的数据进行对齐，导致类别间分布没有很好地分离。

+ DANN-DA 

<img src="images/DANN-DA%20with%20domain%20classifier.png" alt="DANN-DA" width="50%">

和 DANN without domain classifier 差不多， 也是 class 2 分类得比较好， 仍然是比较混乱的特征分布。

+ DANN-DG

<img src="images/DANN-DG%20with%20domain%20classifier.png" alt="DANN-DG" width="50%">

不同类别的特征分布明显分离，分布模式更加清晰。
可能是因为 DG 模式下的 DANN 模型通过对多个源领域的数据进行对齐，学习到更具泛化能力的特征。
从而使得模型具有更强的适应性，类别间的可分性提高。

## 遇到的困难

1. 我发现 DANN 如果启用 Domain Classifier, loss 是不能简单写成 label loss - alpha * domain loss的， 这样前向计算就有问题。 要通过添加 Gradient Reversal Layer 的方式来实现，然后 loss 写成 label loss + domain loss。我最开始是写成了前者，然后 loss 越变越大无法收敛， 最后的识别效果和随机猜差不多， 后来修改了才变得正常了。