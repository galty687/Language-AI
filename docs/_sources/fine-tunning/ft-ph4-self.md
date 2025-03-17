# 使用自监督学习微调Phi4

## 自监督学习

掩码语言模型（Masked Language Model，MLM）的微调主要通过在特定领域的数据上训练模型，使其熟悉该领域的用词习惯和语言模式。​这种训练方式会调整模型的内部参数，从而影响模型对下一个词的预测概率。



## 练习

- 适用于让 Phi **学习微软文档的语言风格**，无需人工标注。
- 方法：Masked Language Modeling（MLM）、Next Sentence Prediction（NSP）。



## 掩码语言模型（Masked Language Model，MLM）

微调过程中，输入文本中的 `[MASK]` 标记通常由微调程序自动生成。具体而言，微调程序会按照预设的策略，对输入文本进行处理，随机选择部分词汇进行掩码操作，以训练模型预测被掩码的词汇。

**掩码策略**

1. **掩码比例**：通常选择输入序列中约15%的词汇进行掩码。
2. **掩码方式**：
   - **80%** 的情况下，将选定的词替换为 `[MASK]` 标记。
   - **10%** 的情况下，保持选定的词不变。
   - **10%** 的情况下，将选定的词替换为词汇表中的随机词。

这种策略的目的是使模型不仅能够学习到 `[MASK]` 标记的上下文，还能在面对未被掩码的词或随机替换的词时，增强模型的鲁棒性。在微调过程中，这些掩码操作由程序自动执行，无需人工干预。开发者只需提供原始文本数据，微调程序会根据上述策略自动生成包含 `[MASK]` 标记的训练数据，以训练模型预测被掩码的词汇。

### 示例数据

```json
[
  {
    "id": "doc1",
    "title": "Introduction",
    "content": "Java is one of the most used programming languages, according to Stack Overflow and GitHub. Java Virtual Machine (JVM) offers a mature way to run Java applications efficiently. Azure offers various ways to deploy your Java applications. No matter what types of Java applications you're running, Azure has a solution. You can choose from batch processes, nanoservices, and microservices, all the way up to Java Enterprise Edition (EE) and Jakarta EE applications. In this module, we look at Java's powerful features and give an overview of Azure deployment offers. This module is for Java developers and system administrators who have experience with running Java applications. There's no coding involved in this conceptual module. Learning objectives By the end of this module, you'll be able to: Differentiate between types of Java applications. Explain the opportunities for Java developers on Azure. Prerequisites Basic development experience in Java or system operating knowledge for Java-based architectures."
  },
  {
    "id": "doc2",
    "title": "Java at Microsoft",
    "content": "Developers from around the world learn programming with Java, and it remains one of the most used languages among enterprises. It can help you solve business requirements at all levels. With millions of Java developers worldwide, Java's success speaks for itself. Java is a strategic language of choice on Azure. We support Java developers on multiple levels to deploy their Java applications. No matter what your architecture looks like, Azure has a solution for you; from monolithic applications to microservices or even serverless applications. Microsoft has a high interest in supporting Java and Java on Azure. Did you know that Microsoft is an official contributor to OpenJDK? Microsoft uses Java in many of its products, like LinkedIn, Yammer, Minecraft, and Surface Duo."
  }
]

```


### 示例代码

在使用 Hugging Face 的 Transformers 库进行 MLM 训练时，关键在于数据的预处理。这通常`DataCollatorForLanguageModeling` 来实现，该类负责在训练时对输入数据进行随机掩码操作。

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,  # 启用掩码语言模型
    mlm_probability=0.15  # 设置掩码的概率
)

```




在上述代码中，`mlm=True` 表示启用掩码语言模型，`mlm_probability=0.15` 指定了掩码的概率，即随机选择 15% 的词汇进行掩码处理。然后，将此 `data_collator` 传递给 `Trainer`，如下所示：

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./phi-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,  # 传入数据整理器
    train_dataset=dataset["train"],
)

```

这样，`Trainer` 在训练过程中会自动应用掩码策略。

MLM通过在输入文本中随机掩盖部分词汇，并要求模型根据上下文预测这些被掩盖的词汇来进行训练。在微调阶段，使用特定领域的数据进行训练，模型会学习到该领域的语言特征和用词习惯。这使得模型在处理该领域的文本时，能够更准确地预测词汇，提高生成文本的质量和相关性。因此，MLM微调通过让模型适应特定领域的语言模式，增强了模型在该领域的表现能力。



## 下一句预测（NSP）设置

对于 NSP 任务，需要在数据预处理阶段构造句子对，并为每对句子指定标签（即第二个句子是否为第一个句子的后续句）。这通常需要自定义数据集，并在模型定义时选择支持 NSP 任务的模型架构，例如 `BertForPreTraining`。以下是一个简要的示例：

```python
from transformers import BertForPreTraining, Trainer, TrainingArguments

model = BertForPreTraining.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="./bert-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=custom_nsp_dataset,  # 自定义的 NSP 数据集
)

```

在此示例中，`custom_nsp_dataset` 是一个包含句子对和对应标签的数据集，`BertForPreTraining` 模型同时支持 MLM 和 NSP 任务。