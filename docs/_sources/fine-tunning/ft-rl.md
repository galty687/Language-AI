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

#### 0 安装库

```
!pip install datasets        # for `from datasets import load_dataset`
!pip install trl             # for `from trl import DPOConfig, DPOTrainer`
!pip install transformers    # for `from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline`
!pip install torch           # for `import torch`

```




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



各参数含义：

| 参数名                          | 含义说明                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| **output_dir**                  | 模型训练结果（包括 checkpoints）的输出路径。<br/>训练完成后的模型会保存在这个目录。 |
| **logging_steps**               | 每隔多少步记录一次训练过程中的日志（如 loss 值、学习率等）。<br/>此处设为25，即每25步输出一次日志。 |
| **per_device_train_batch_size** | 每个GPU（或设备）用于训练的批次大小（batch size）。<br/>此处为每个设备8个样本。 |
| **per_device_eval_batch_size**  | 每个GPU（或设备）用于评估时的批次大小（batch size）。<br/>此处为8个样本，与训练批次大小一致。 |
| **num_train_epochs**            | 训练的总轮数（epochs）。<br/>此处设为3，表示数据集整体被遍历三次。 |
| **load_best_model_at_end**      | 训练结束后是否自动加载性能最优的模型。<br/>这里设为`True`，会自动加载在验证集上表现最好的checkpoint。 |
| **metric_for_best_model**       | 选择最佳模型的指标。<br/>这里使用的是`eval_loss`，表示以验证集上的loss作为评判标准。 |
| **save_strategy**               | 模型的保存策略。<br/>这里为`"epoch"`，表示在每个epoch结束时保存一次checkpoint。 |
| **eval_strategy**               | 模型的评估策略。<br/>这里为`"epoch"`，即每个epoch结束时进行一次验证。 |
| **eval_steps**                  | 执行评估时的步数间隔（仅当评估策略为`steps`时有效）。<br/>由于这里设置的是`epoch`，此参数并不生效，但通常设置为步数间隔进行中间验证时使用。 |

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

运行后的输出：

```
Extracting prompt in train dataset: 100%|██████████| 1026/1026 [00:00<00:00, 12014.88 examples/s]
Applying chat template to train dataset: 100%|██████████| 1026/1026 [00:00<00:00, 8241.62 examples/s]
Tokenizing train dataset: 100%|██████████| 1026/1026 [00:00<00:00, 3414.80 examples/s]
Extracting prompt in eval dataset: 100%|██████████| 114/114 [00:00<00:00, 11163.40 examples/s]
Applying chat template to eval dataset: 100%|██████████| 114/114 [00:00<00:00, 7319.34 examples/s]
Tokenizing eval dataset: 100%|██████████| 114/114 [00:00<00:00, 3080.27 examples/s]
```

**Progress:** `[387/387 13:18, Epoch 3/3]`

| Epoch | Training Loss | Validation Loss | Rewards/chosen | Rewards/rejected | Rewards/accuracies | Rewards/margins | Logps/chosen | Logps/rejected | Logits/chosen | Logits/rejected |
| ----- | ------------- | --------------- | -------------- | ---------------- | ------------------ | --------------- | ------------ | -------------- | ------------- | --------------- |
| 1     | 0.560100      | 0.562748        | 2.503206       | 1.936790         | 0.658333           | 0.566416        | -31.019157   | -40.121326     | -3.392188     | -3.384011       |
| 2     | 0.408400      | 0.520122        | 1.082610       | -0.059237        | 0.766667           | 1.141847        | -45.225121   | -60.081593     | -3.431983     | -3.411697       |
| 3     | 0.286200      | 0.584226        | 0.153396       | -1.310745        | 0.725000           | 1.464142        | -54.517258   | -72.596687     | -3.212612     | -3.183683       |

#### 6 使用微调后的模型

```python
# Load the fine-tuned model
ft_model = trainer.model

# Set up text generation pipeline
generator = pipeline("text-generation", model=ft_model, tokenizer=tokenizer, device='mps')

# Example prompt
prompt = format_chat_prompt(dataset['valid']['prompt'][0][0]['content'])

# Generate output
outputs = generator(prompt, max_length=100, truncation=True, num_return_sequences=1, temperature=0.7)

print(outputs[0]['generated_text'])
```

输出：

```markdown
Device set to use mps
/Users/zhijungao/ENTER/lib/python3.10/site-packages/transformers/pytorch_utils.py:328: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  test_elements = torch.tensor(test_elements)
<|im_start|>user
Given the YouTube video idea write an engaging title.

**Video Idea**: intro independent component analysis

**Additional Guidance**:
- Title should be between 30 and 75 characters long
- Only return the title idea, nothing else!<|im_end|>
<|im_start|>assistant
Independent Component Analysis for Beginners
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
3. 电子书： [Reinforcement Learning from Human Feedback](https://rlhfbook.com/)
