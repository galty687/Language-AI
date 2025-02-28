# 文本分类



## 任务

电影评价的情感分析

数据集：[rotten_tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)



## 准备数据

1. 安装Hugging Face datasets

```python
!pip install datasets
```

2. 加载数据集

```python
from datasets import load_dataset
# Load our data
data = load_dataset("rotten_tomatoes")

```

输出：

```python
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
})
```

3. 查看训练集

   ```
   data["train"][0,-1]
   ```

   输出：

   ```python
   {'text': ['the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
     'things really get weird , though not particularly scary : the movie is all portent and no content .'],
    'label': [1, 0]}
   ```

   

## 模型选择

选择模型是一件复杂的事情，在Hugging Face上有6万多个文本分类模型，一般来说可以从如下几个方面来评估：



1. 明确任务需求。首先确定你的具体任务，比如情感分析、垃圾邮件检测或主题分类，并明确对模型性能、速度和资源消耗的要求。
2. 查看模型描述与文档。详细阅读模型页面上的描述、训练数据、使用场景以及相关论文。了解模型是如何预训练和微调的，以及它是否适合你的领域。
3. 参考性能指标。检查模型在公开基准数据集上的表现（如准确率、F1 分数等），选取那些在与你任务相似的数据集上表现较好的模型。

4. 关注用户反馈和社区讨论。在 Hugging Face 的模型页面中查看用户评价和讨论，可以帮助你了解模型在实际使用中的优缺点和潜在问题。
5. 考虑模型大小和部署要求。根据你的硬件条件和部署场景选择合适的模型大小。较大的模型可能性能更好，但资源消耗也更高；而较小的模型则更适合资源受限的场景。
6. 试验与验证。在有限的数据集上先进行试验，验证模型是否满足你的预期效果，再决定是否进行大规模部署和进一步的微调。







## 使用表示模型

本任务重选择：Twitter-RoBERTa-base for Sentiment Analysis 

该模型基于 RoBERTa-base 架构，专门针对 Twitter 数据中的情感分析任务进行了微调。该模型利用大量 Twitter 语料库训练，能够较好地捕捉推文中常见的非正式语言、缩写、表情符号和标签，从而对文本情感（如正面、负面或中性）进行准确分类。



1. 加载模型

   ```python
   from transformers import pipeline
   # Path to our HF model
   model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
   # Load model into pipeline
   pipe = pipeline(
   model=model_path,
   tokenizer=model_path,
   return_all_scores=True,
   device="cuda:0"
   )
   ```

   

2. 执行分类任务

   ```python
   import numpy as np
   from tqdm import tqdm
   from transformers.pipelines.pt_utils import KeyDataset
   from sklearn.metrics import classification_report
   
   # Run inference
   y_pred = []
   for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
       negative_score = output[0]["score"]
       positive_score = output[2]["score"]
       assignment = np.argmax([negative_score, positive_score])
       y_pred.append(assignment)
   ```

   

3. 评估模型性能

   ```python
   def evaluate_performance(y_true, y_pred):
       """Create and print the classification report"""
       performance = classification_report(
           y_true, y_pred,
           target_names=["Negative Review", "Positive Review"]
       )
       print(performance)
   ```

   

评估结果：

```md
                 precision    recall  f1-score   support

Negative Review       0.76      0.88      0.81       533
Positive Review       0.86      0.72      0.78       533

       accuracy                           0.80      1066
      macro avg       0.81      0.80      0.80      1066
   weighted avg       0.81      0.80      0.80      1066
```



## 使用生成式模型（1）



1. 加载模型

   ```python
   pipe = pipeline(
   "text2text-generation",
   model="google/flan-t5-small",
   device="cuda:0"
   )
   ```

   

2. 准备数据

   ```python
   prompt = "Is the following sentence positive or negative? "
   data = data.map(lambda example: {"t5": prompt + example['text']})
   data
   ```

   

输出：

```

```



3. 开始提取

   ```python
   # Run inference
   y_pred = []
   for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
       text = output[0]["generated_text"]
       y_pred.append(0 if text == "negative" else 1)
   ```

   

4. 评估效果

   ```python
   evaluate_performance(data["test"]["label"], y_pred)
   ```

   

输出：



```md
                 precision    recall  f1-score   support

Negative Review       0.83      0.85      0.84       533
Positive Review       0.85      0.83      0.84       533

       accuracy                           0.84      1066
      macro avg       0.84      0.84      0.84      1066
   weighted avg       0.84      0.84      0.84      1066
```



F1 得分为 0.84，表明 Flan-T5 这类生成式模型的强大能力。



## 使用生成式模型（2）

1. 创建openai client

   ```python
   import openai
   # Create client
   client = openai.OpenAI(api_key="YOUR_KEY_HERE")
   ```

   

2. 创建生成函数

   ```python
   def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
       """Generate an output based on a prompt and an input document."""
       messages = [
           {
               "role": "system",
               "content": "You are a helpful assistant."
           },
           {
               "role": "user",
               "content": prompt.replace("[DOCUMENT]", document)
           }
       ]
       chat_completion = client.chat.completions.create(
           messages=messages,
           model=model,
           temperature=0
       )
       return chat_completion.choices[0].message.content
   ```

   

3. 创建提示词

   ```python
   # Define a prompt template as a base
   prompt = """Predict whether the following document is a positive or negative
   movie review:
   [DOCUMENT]
   If it is positive return 1 and if it is negative return 0. Do not give any
   other answers.
   """
   # Predict the target using GPT
   document = "unpretentious , charming , quirky , original"
   chatgpt_generation(prompt, document)
   ```

   

4. 分类

   ```python
   predictions = [
   chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]["text"])
   ]
   ```

   

5. 查看分类结果

   ```python
   y_pred = [int(pred) for pred in predictions]
   ```

   

6. 评估性能

   ```
   y_pred = [int(pred) for pred in predictions]
   evaluate_performance(data["test"]["label"], y_pred)
   ```

   

输出：

```python
                 precision    recall  f1-score   support

Negative Review       0.85      0.97      0.90       533
Positive Review       0.96      0.83      0.89       533

       accuracy                           0.90      1066
      macro avg       0.91      0.90      0.90      1066
   weighted avg       0.91      0.90      0.90      1066
```



### 使用GPT最新模型 

模型：` gpt-4o-mini-2024-07-18`

结果：

```python
                 precision    recall  f1-score   support

Negative Review       0.88      0.96      0.92       533
Positive Review       0.96      0.87      0.91       533

       accuracy                           0.91      1066
      macro avg       0.92      0.91      0.91      1066
   weighted avg       0.92      0.91      0.91      1066
```



模型：`gpt-4o-2024-08-06`

结果：

```python
                 precision    recall  f1-score   support

Negative Review       0.88      0.97      0.92       533
Positive Review       0.97      0.86      0.91       533

       accuracy                           0.92      1066
      macro avg       0.92      0.92      0.92      1066
   weighted avg       0.92      0.92      0.92      1066
```

