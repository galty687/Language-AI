# 基于HF Transformer的NLP实践的练习答案

## 练习1答案

```python
from transformers import pipeline

# 1. 加载模型和分词器
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_analysis = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

# 2. 输入文本
text = "I absolutely love this new phone!"

# 3. 进行情感分析
result = sentiment_analysis(text)

# 4. 输出结果
print("情感分析结果:", result)

```

输出：

```
情感分析结果: [{'label': '5 stars', 'score': 0.962716281414032}]
```

> 模型认为该句子表达了非常正面的情感，其中 '5 stars' 表示最高情感评级，得分 0.962 表示模型对此预测的置信度为 96.27%。



## 练习2 答案

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. 加载模型和分词器
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 准备输入
text = "I absolutely love this new phone!"
inputs = tokenizer(text, return_tensors="pt")

# 3. 模型推理，获取 logits
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 4. 将 logits 转换为概率
probs = torch.softmax(logits, dim=-1)

# 5. 找到概率最大的标签
pred_label_idx = torch.argmax(probs, dim=-1).item()
label_names = model.config.id2label
pred_label = label_names[pred_label_idx]

# 输出结果
print("Logits:", logits)
print("Probabilities:", probs)
print("Predicted label index:", pred_label_idx)
print("Predicted label:", pred_label)

```

输出：

```python
Logits: tensor([[-2.0808, -2.6891, -1.3933,  0.9046,  4.3200]])
Probabilities: tensor([[1.5983e-03, 8.6986e-04, 3.1786e-03, 3.1637e-02, 9.6272e-01]])
Predicted label index: 4
Predicted label: 5 stars
```



### 解释：

1. 加载模型和分词器

   - **model_name**：指定了要加载的模型名称。在这里，使用的是一个多语言的 BERT 模型，专门用于情感分析任务。

   - **model = ...**：通过 `from_pretrained` 方法加载预训练的情感分类模型，该模型已经在大量数据上进行过训练，能对文本情感进行判断。

   - **tokenizer = ...**：加载与模型对应的分词器，确保输入文本的编码格式与模型一致。

2. 模型推理，获取 logits

   logits 是模型输出的原始值，它是每个类别的得分（score），**尚未进行概率转换**（未经过softmax函数处理），每个类别的logit越大，表示模型越倾向预测为对应的标签。

   

   > 举个简单的例子： 假设模型预测三个类别（积极、中性、消极）， logits 为 `[3.2, -1.1, 0.5]` 表示模型更倾向于第一个类别（积极）。

   

   - **with torch.no_grad()**：在推理过程中禁用梯度计算，这样可以节省内存并提高推理速度，因为这里不需要反向传播计算梯度。

   - **outputs = model (inputs)**：将编码后的输入数据传入模型，得到模型的输出。

   - **logits = outputs.logits**：从输出中提取 logits，logits 是模型在各个类别上的原始预测分数（未归一化）。

     

3. 将 logits 转换为概率
    - **torch.softmax**：对 logits 进行 softmax 操作，转换成概率分布。这里的 `dim=-1` 表示在最后一个维度上应用 softmax（即每个样本在所有类别上的概率和为 1）

      通过分类头输出 logits：

      - 比如 `[1.1, 0.3, -0.5, 0.2, 4.2]`

      softmax后转化为概率：

      - 比如 `[0.04, 0.02, 0.01, 0.03, 0.91]`

4. 找到概率最大的标签

    - torch.argmax(probs, dim=-1)：找到概率最高的类别索引，代表模型的预测结果。调用 .item() 将单个张量转换为 Python 的数值类型。

    - model.config.id2label：从模型配置中获取索引到标签的映射字典。

    - pred_label：利用映射字典，将预测的索引转换为具体的标签名称（例如情感等级）。