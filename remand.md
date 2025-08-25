# 灾难推文分类项目

## 概述

本项目旨在构建一个机器学习模型，用于判断一条推文是否在描述真实的灾难事件（如地震、火灾、洪水），而不是比喻、玩笑或日常用语。项目源自Kaggle上的[“Natural Language Processing with Disaster Tweets”](https://www.kaggle.com/c/nlp-getting-started)竞赛。该数据集包含约10,000条人工标注的推文。

## 项目结构

- `train.csv`: 训练数据，包含`id`, `text`, `target`等字段。`target`为1表示是灾难推文，为0则表示不是。
- `test.csv`: 测试数据，包含`id`和`text`，用于生成预测结果。
- 主代码文件：包含数据清洗、模型构建、训练和预测的全部代码（例如：`disaster_tweets_classification.ipynb` 或 `.py` 文件）。

## 技术栈与依赖

- **编程语言:** Python 3
- **核心库:**
  - `pandas`: 数据处理和分析
  - `numpy`: 数值计算
  - `tensorflow` / `keras`: 构建和训练深度学习模型
  - `scikit-learn`: 数据分割、评估指标（F1 Score）
  - `re`: 正则表达式，用于文本清洗

## 如何运行

1. **环境配置：** 确保已安装所需的Python库。可以使用以下命令安装：

   bash

   ```
   pip install pandas numpy tensorflow scikit-learn
   ```

2. **准备数据：** 将Kaggle竞赛提供的`train.csv`和`test.csv`文件放置在代码同一目录下（或根据代码中的路径修改）。

3. **执行代码：** 运行整个代码脚本或Notebook。代码将按顺序执行以下步骤：

   - 加载训练和测试数据。
   - 对推文文本进行清洗（小写化、移除URL、移除@提及、移除非字母数字字符）。
   - 使用Tokenizer将文本转换为整数序列，并进行填充以保证输入长度一致。
   - 构建一个嵌入层(Embedding) + 双向LSTM (Bidirectional LSTM) 的深度学习模型。
   - 将训练集划分为训练集和验证集。
   - 训练模型，并在验证集上计算F1分数以评估性能。
   - 对测试集进行预测，并生成名为`submission.csv`的Kaggle提交文件。

## 模型详情

- **架构：**
  - **嵌入层 (Embedding Layer):** 将整数序列转换为密集向量表示。
  - **双向LSTM层 (Bidirectional LSTM):** 能够同时从前往后和从后往前学习文本的上下文信息，更好地捕捉语义。
  - **全连接层 (Dense Layer) 与 Dropout:** 添加非线性并防止过拟合。
  - **输出层 (Output Layer):** 使用Sigmoid激活函数，输出一个介于0和1之间的概率值。
- **损失函数:** 二元交叉熵 (Binary Crossentropy)
- **优化器:** Adam
- **评估指标:**
  - **训练时监控:** 准确率 (Accuracy)
  - **最终验证指标:** F1分数 (F1 Score)，该指标更适合处理可能存在的类别不平衡问题。

## 结果与改进

- 运行代码后，控制台将输出模型在验证集上的F1分数。这是评估模型性能的关键指标。
- **改进方向：**
  - **文本预处理:** 尝试更精细的清洗（如处理表情符号、词形还原）、使用更先进的文本表示（如BERT预训练模型）。
  - **模型架构:** 尝试不同的模型（如CNN、Transformer）、调整超参数（嵌入维度、LSTM单元数、学习率）、增加模型深度。
  - **正则化:** 引入早停（Early Stopping）、L2正则化等进一步防止过拟合。
  - **处理不平衡:** 若数据不平衡，可在训练时使用类别权重或采用过采样/欠采样技术。