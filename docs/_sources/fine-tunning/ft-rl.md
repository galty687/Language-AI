# 基于人类反馈微调大模型（RLHF + DPO）

## 引言
大语言模型（LLMs）可以执行各种自然语言处理（NLP）任务，但其初始训练数据来源广泛，可能导致输出内容与人类期望不一致。因此，需要进一步微调，使其更符合人类偏好。本教程介绍如何使用强化学习（RLHF）和直接策略优化（DPO）对大模型进行微调。

## RLHF 概述
### 什么是 RLHF？
强化学习（RL）是一种通过试错方式进行学习的训练方法。RLHF（Reinforcement Learning from Human Feedback）指的是通过人类反馈训练奖励模型（Reward Model），再用该奖励模型优化 LLM，使其输出更符合人类偏好。

### 奖励模型的训练
1. 生成多个候选响应。
2. 由人工标注员根据特定准则对这些响应进行排序。
3. 用这些排序数据训练奖励模型，使其能评估模型生成的文本质量。

### Proximal Policy Optimization（PPO）
在 RLHF 过程中，常用的优化算法是 Proximal Policy Optimization（PPO）。其目标是最大化奖励模型的分数，同时限制模型更新的幅度，以保证稳定性。

### RLHF 的局限性
RLHF 受到训练数据质量的限制，最终性能取决于奖励模型的准确性。此外，其训练流程复杂，需要大量计算资源。

## 直接策略优化（DPO）
### DPO 的基本概念
DPO（Direct Policy Optimization）是一种无需 RL 过程即可优化模型的替代方案。DPO 通过将 RLHF 重新表述为一个文本分类任务，直接调整模型的概率分布。

### DPO 的优点
- 训练流程更简单，无需额外的奖励模型。
- 计算开销更小。
- 能获得与 RLHF 相似的性能提升。

## 使用 DPO 微调 Qwen2.5–0.5B-Instruct

### 3.1 数据准备
为了微调模型，我们需要一个偏好数据集。数据准备流程如下：
1. 生成 114 个视频创意。
2. 使用 Qwen2.5–7B-Instruct 生成 5 个标题。
3. 创建 10 组头对头标题对比（5 选 2 组合）。
4. 手动标注 1140 组标题对，选择更优标题。
5. 格式化数据，使其包含 `prompt`、`chosen` 和 `rejected` 三列。

### 代码实现
#### 1 导入必要的库
```markdown
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
```

#### 2 加载数据集和模型
```python
# 加载数据集
dataset = load_dataset("shawhin/youtube-titles-dpo")

# 加载预训练模型和分词器
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

#### 3 生成示例标题
```python
# 格式化用户输入
def format_chat_prompt(user_input):
    return f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

# 设置文本生成管道
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 示例输入
prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])
outputs = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.7)
print(outputs[0]['generated_text'])
```

#### 4 设置 DPO 训练参数
```python
ft_model_name = model_name.split('/')[1].replace("Instruct", "DPO")
training_args = DPOConfig(
    output_dir=ft_model_name,
    logging_steps=25,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=1,
)
```

#### 5 训练模型
```python
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['valid'],
)
trainer.train()
```

### 评估微调后的模型
为了评估微调效果，我们使用以下步骤：
1. 选取 50 个随机视频创意。
2. 用基础模型和微调模型分别生成标题。
3. 人工对比标题对，标注偏好。
4. 计算微调模型生成的标题被偏好的比例。

最终结果表明，微调后的模型标题在 68% 的情况下被认为更优。

## 结论
- RLHF 通过强化学习优化模型，但训练复杂，计算成本高。
- DPO 通过重构 RLHF 任务，避免了强化学习过程，使训练更简单高效。
- 在本教程中，我们使用 DPO 微调了 Qwen2.5–0.5B-Instruct，使其在 YouTube 标题生成任务中更符合人类偏好。

如果你对该方法感兴趣，欢迎尝试使用不同的数据集和模型进行实验！



## 参考

1. [示例代码](https://github.com/ShawhinT/YouTube-Blog/blob/main/LLMs/dpo/1-generate_synthetic_titles.ipynb)
2. [数据集](https://huggingface.co/datasets/shawhin/youtube-titles-dpo)
