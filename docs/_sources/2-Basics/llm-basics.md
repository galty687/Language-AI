# LLM 基础



## transformers

Hugging Face 的 transformers 是一个开源库，主要用于自然语言处理（NLP）和其他基于 Transformer 架构的任务。它的主要特点包括：

•	**丰富的预训练模型**：支持 BERT、GPT、T5、RoBERTa 等多种主流模型，方便直接应用于文本分类、生成、翻译等任务。

•	**统一的接口**：提供统一的 API 来加载模型和分词器，简化数据预处理和模型推理过程。

•	**跨框架支持**：同时兼容 PyTorch 和 TensorFlow，用户可根据习惯和需求选择合适的深度学习框架。

•	**易于微调**：用户可以基于预训练模型进行微调，快速适应特定任务或领域应用。



安装

```bash
pip install transformers
```



异常即其他情形，可参考：[官方文档](https://huggingface.co/docs/transformers/v4.49.0/en/installation?install=pip)



### AutoModelForCausalLM

AutoModelForCausalLM 是 Hugging Face transformers 库中提供的一个便捷类，用于自动加载适用于因果语言建模（Causal Language Modeling）任务的模型。其主要功能包括：

1.	**自动推断模型架构**

根据你提供的预训练模型名称或路径，AutoModelForCausalLM 会读取模型的配置文件，并自动确定适合的模型架构（如 GPT、GPT-2 等），无需手动指定具体模型类型。

2.	**文本生成任务支持**

因果语言模型主要用于生成连续文本，例如自动续写、对话生成等任务。加载这样的模型后，可以利用它进行自然语言生成。

3.	**简化模型加载流程**

通过自动推断和加载预训练权重、配置等，开发者只需要知道模型名称即可快速加载并使用模型，降低了入门和使用复杂性。



### AutoTokenizer

AutoTokenizer 是 Hugging Face transformers 库中的一个便捷类，它主要负责为预训练模型自动加载和配置相应的分词器。具体功能包括：

1.	**文本分词**。将输入的自然语言文本拆分成模型能够理解的 token（词元或子词单元），这一步是模型处理文本的前置步骤。

2.	**自动匹配模型**。根据你提供的预训练模型名称或路径，AutoTokenizer 会自动选择与模型相匹配的分词器，保证文本编码方式与模型训练时使用的方式一致。

3.	**编码和解码**。除了将文本转换为 token id 序列（编码），它还能将模型输出的 token id 序列解码回可读的文本（解码），方便理解和展示生成结果。



## 下载和运行LLM

### Microsoft Phi

Microsoft 的 Phi 模型是一系列由微软研究团队开发的小型语言模型（SLMs），其目标是在参数规模较小的前提下提供与大型语言模型相媲美的性能。Phi 模型的发展经历了从 Phi-1、Phi-1.5、Phi-2 到最新的 Phi-3，不断提升语言理解、推理、编码等多方面能力。其中，Phi-3 系列包括多个版本，如 Phi-3-mini、Phi-3-small 和 Phi-3-medium，参数规模分别约为 3.8 亿、7 亿和 14 亿。经过指令微调后，这些模型能够更好地理解和执行用户指令，生成高质量且安全的回答。此外，得益于轻量化设计，Phi 模型适用于资源受限的设备，如移动端和边缘计算场景，提供了低成本、高效率的生成式 AI 解决方案。

第三代 Phi-3 系列模型，这一代包含三个不同规模的版本：

- **Phi-3-mini**：约 3.8 亿参数（38亿参数），默认上下文长度为 4K，适合在资源受限的设备（如手机）上运行。
- **Phi-3-small**：约 7 亿参数（70亿参数），支持更长的上下文（通常为 8K），在多语言和理解能力上有所提升。

- **Phi-3-medium**：约 14 亿参数（140亿参数），参数规模更大，性能更强，但对硬件要求也更高。

模型名称中的 “instruct” 表示该模型经过了指令微调，也就是通过使用大量标注了“指令–回答”对的数据对模型进行了额外训练，从而使模型能够更好地理解和执行用户给出的指令，生成更符合预期、更有针对性的回答。与普通的基础语言模型相比，“instruct” 版本在回答问题时通常会更“听话”，输出也更加安全和实用。



### 加载模型和分词器

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
"microsoft/Phi-3-mini-4k-instruct",
device_map="cuda",
torch_dtype="auto",
trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
```

### 让模型输出

```python
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap.Explain how it happened.<|assistant|>"
# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate the text
generation_output = model.generate(
input_ids=input_ids,
max_new_tokens=20
)

# Print the output
print(tokenizer.decode(generation_output[0]))
```



模型输出结果：

```md
Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|> Subject: Sincere Apologies for the Gardening Mishap


Dear
```



查看`input_ids`中的内容

```python
tensor([[14350,385,4876,27746,5281,304,19235,363,278,25305,293,16423,292,286,728,481,29889,12027,7420,920,372,9559,29889,32001]], device='cuda:0')
```



将id转为token

```python
for id in input_ids[0]:
	print(tokenizer.decode(id))
```



```md
Write
an
email
apolog
izing
to
Sarah
for
the
trag
ic
garden
ing
m
ish
ap
.
Exp
lain
how
it
happened
.
<|assistant|>
```

查看结果发现：

- 一些 token 是完整的单词（例如：Write、an、email）。

- 一些 token 是单词的一部分（例如：apolog、izing、trag、ic、m、ish、ap）。

- 标点符号则作为独立的 token。



检查输出

```python
print(generation_output)
```



输出结果的ID：



```python
tensor([[14350,385,4876,27746,5281,304,19235,363,278,25305,293,16423,292,286,728,481,29889,9544,7420,920,372,9559,29889,32001,3323,622,29901,317,3742,406,6225,11763,363,278,19906,292,341,728,481,13,13,13,29928,799]], device='cuda:0')


```

检查Token 32001之后的内容

```
print(tokenizer.decode(3323))
print(tokenizer.decode(622))
print(tokenizer.decode(29901))
print(tokenizer.decode(317))
print(tokenizer.decode(3742))
print(tokenizer.decode(406))
```

输出结果：



```markdown
Sub
ject
:
S
inc
ere
```



Phi 模型的tokenizer

1.	词汇表（Vocabulary）
分词器在训练时会构建一个词汇表，其中每个词或子词都对应一个唯一的整数 ID。
2.	分词过程
当输入文本传入分词器时，它会根据预定的编码算法（例如 Byte-Pair Encoding（BPE）或 SentencePiece）将文本拆分成多个 token。其中，有些 token 可能代表完整的单词，而有些则是单词的子部分。
3.	特殊标记
分词器还会在文本的开头或结尾添加特殊标记（例如  <s>  表示开始，</s>  表示结束），这些特殊标记也会对应特定的 token ID。
4.	编码为整数
最终，文本中的每个 token 都被转换为一个整数 ID，形成一个整数序列，也就是 token ID 序列。这些整数序列就是送入模型进行计算的输入。



查看Phi的词汇表

```python
vocab = tokenizer.get_vocab()
print(vocab)
```



```
vocab = tokenizer.get_vocab()
# 按照 token ID 排序，生成一个 (token, token_id) 的列表
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
# 打印前 100 个词汇
print(vocab_sorted[:100])
```



输出结果：

```
{'小': 30446, '▁SY': 28962, '▁earnest': 24828, '▁minimum': 9212, '▁sugar': 26438, 'Cam': 14353, '▁build': 2048, '▁agr': 9221, 'ierten': 12025, 'emet': 21056, 'uuid': 25118, '▁TRUE': 15676, '▁notification': 12519, '▁inside': 2768, '▁extens': 21103, '▁Wür': 21241, '▁gross': 22683, 'inf': 7192, 'Μ': 30362, "'],": 7464, 'bek': 16863, 'Values': 9065, 'ón': 888, 'три': 7678, 'шти': 12316, '▁Bush': 24715, '▁decom': 17753, '▁kommen': 28171, '.$': 7449, 'DOC': 28665, '▁mang': 25016, 
```



## 测试GPT2

查看模型形状

```python
from transformers import GPT2Tokenizer, GPT2Model

# 加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# 直接访问词嵌入层
word_embeddings = model.wte.weight  # 形状通常为 (vocab_size, embedding_dim)
print(word_embeddings.shape)  # 比如 (50257, 768)
```



获取某个 token 的词向量



```python
token = "hello"
token_id = tokenizer.convert_tokens_to_ids(token)
embedding = word_embeddings[token_id]
print(embedding)
```



## Contextual Embeddings

查看Bank在不同语境下的embeddings

```
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # 切换到评估模式

sentences = [
    "I went to the bank to deposit money.",
    "The river bank was covered with lush vegetation."
]

for sentence in sentences:
    # 编码句子，返回 PyTorch 张量
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取最后一层隐藏状态，形状为 (batch_size, seq_length, hidden_dim)
    token_embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_length, hidden_dim)
    # 将 token IDs 转换回 token 字符串
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze(0))
    # 找出单词 "bank" 的位置（注意：BERT 将 "bank" 分词为单个 token "bank"）
    bank_indices = [i for i, token in enumerate(tokens) if token == "bank"]
    if bank_indices:
        bank_embedding = token_embeddings[bank_indices[0]]
        # 为了展示，我们只显示前 10 个数值（真实的嵌入向量通常是768维）
        print(f"句子: {sentence}")
        print(f"token 列表: {tokens}")
        print(f"‘bank’ 的嵌入向量（前10个数值）: {bank_embedding[:10].numpy()}\n")
```

输出：

```md
句子: I went to the bank to deposit money.
token 列表: ['[CLS]', 'i', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.', '[SEP]']
‘bank’ 的嵌入向量（前10个数值）: [ 0.7091013  -0.25904247 -0.01858949 -0.09361451  1.2636592   0.02228517
 -0.30962497  0.9713595  -0.10284916  0.20124747]

句子: The river bank was covered with lush vegetation.
token 列表: ['[CLS]', 'the', 'river', 'bank', 'was', 'covered', 'with', 'lush', 'vegetation', '.', '[SEP]']
‘bank’ 的嵌入向量（前10个数值）: [-0.17602193 -0.55783457 -0.23129024 -0.1350407  -0.3741462   0.35257423
 -0.04954641  1.3635753   0.208523   -0.48707223]
```



## LM Head

```python
prompt = "The capital of France is"
# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
# Tokenize the input prompt
input_ids = input_ids.to("cuda")
# Get the output of the model before the lm_head
model_output = model.model(input_ids)
# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])

print(lm_head_output[0,-1])
```



Top_K

```
import torch
# 获取最后一个位置的 logits
logits = lm_head_output[0, -1]
# 获取前2个最高的值及对应的索引
topk = torch.topk(logits, k=10)
# 第二高的 token 的 ID
second_token_id = topk.indices[9]
# 解码为文本
print(tokenizer.decode(second_token_id))
```

