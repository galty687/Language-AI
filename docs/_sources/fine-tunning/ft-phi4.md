# SFT 微调 Phi-4

使用监督学习的方式，主要素材来自 *Microsoft Manual of Style。*

## 测试模型的改写能力：



1. 加载模型

   ```python
   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
   torch.random.manual_seed(0)
   
   model_path = "microsoft/Phi-4-mini-instruct"
   
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       device_map="auto",
       torch_dtype="auto",
       trust_remote_code=True,
   )
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   
   pipe = pipeline(
       "text-generation",
       model=model,
       tokenizer=tokenizer,
   )
   ```

   

2. 提示词

   ```python
   sent = "Follow these steps in order to change your password."
   
   messages = [
       {"role": "system", "content": "You are a professional content editor with extensive knowledge of the Microsoft Manual of Style. Your task is to edit and improve any content I provide, ensuring it adheres to the Microsoft Manual of Style guidelines. Focus on clarity, grammar, punctuation, and overall readability, while maintaining the original meaning. Follow any additional specific instructions I include in my messages."},
       {"role": "user", "content": sent}
   ]
   
   generation_args = {
       "max_new_tokens": 500,
       "return_full_text": False,
       "temperature": 0.0,
       "do_sample": False,
   }
   
   output = pipe(messages, **generation_args)
   print(output[0]['generated_text'])
   ```

   

3. 模型输出：

   ```md
   Follow these steps to change your password:
   
   1. Go to the login page of the website or application.
   2. Click on the "Forgot Password" or "Can't access your account?" link.
   3. Enter your account information, such as your email address or username, to receive a password reset link.
   4. Check your email inbox for the password reset link and click on it.
   5. Follow the instructions provided in the link to create a new password.
   6. Make sure your new password is strong and unique, and save it in a secure place.
   7. Log in to your account using your new password.
   ```



4. 其他例子：

   ```md
   input: It isn’t terribly difficult to change your password.
   
   output: It isn't terribly difficult to change your password.
   
   input: When you lock down your computer . . .
   output: When you lock down your computer, you are taking steps to secure it and protect your data from unauthorized access. This can involve setting a strong password, enabling a firewall, and installing antivirus software. By doing so, you help ensure the safety and privacy of your personal information.
   ```

   

## 尝试改进提示词

### 全部风格的提示词

```python
sent = "Forgot your password? Provide your secret answer."

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert editor specializing in Microsoft style. Please revise any text you receive "
            "according to official Microsoft style guidelines. In particular, ensure the following:\n\n"
            "1. **Be concise**\n   - Don’t use two or three words when one will do (for example, avoid “in order to” "
            "when “to” suffices).\n   - Keep sentences short and direct.\n\n"
            "2. **Prefer “because” to indicate cause**\n   - Use “because” instead of “as” or “since” to explain why "
            "something happens.\n\n"
            "3. **Use words as defined in the dictionary or style sheet**\n   - Avoid coining new terms unnecessarily.\n   - "
            "Use consistent terminology for the same concept.\n\n"
            "4. **Adhere to correct usage for phrases like “set up” vs. “setup”**\n   - “Setup” is the noun (for example, "
            "“The setup took five minutes.”).\n   - “Set up” is the verb (for example, “Set up your account.”).\n\n"
            "5. **Choose countable vs. non-countable terms carefully**\n   - Use “number” for countable items (e.g., “A "
            "significant number of users...”)\n   - Use “amount” for non-countable items (e.g., “A large amount of data...”)\n\n"
            "6. **Use the imperative voice for instructions**\n   - Address the user directly (e.g., “Select,” “Choose,” "
            "or “Go to”).\n\n"
            "7. **Avoid unnecessary synonyms that might confuse the reader**\n   - Use the same term for a concept throughout "
            "to maintain clarity, especially for technical terms.\n\n"
            "Please transform any text you receive into a version that meets these guidelines. If necessary, break down "
            "run-on sentences, use active voice, and focus on clarity, accuracy, and directness. Return only your edited "
            "version, without additional commentary."
        )
    },
    {
        "role": "user",
        "content": sent
    }
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

输出结果：

```md
Forgot your password? Enter your secret answer.
```

全部实现改写比较难：



### 单一规则：

#### 测试实例：Use questions sparingly. 

提示词：

```python
sent = "Like what you see? Get more nature themes online."

messages = [
    {
        "role": "system",
        "content": "You are an expert writer specializing in Microsoft style. The user will provide text that may contain questions and casual phrasing. Your job is to revise it into a clear, concise, and direct Microsoft style voice. In particular:\n\n1. Avoid overusing questions or rhetorical questions. Focus on providing answers and clear instructions.\n2. Maintain a trustworthy tone by giving guidance rather than inventing questions.\n3. Use the imperative voice when giving instructions (e.g., 'If you forgot your password, provide your secret answer.')\n4. Be concise and eliminate unnecessary words or phrases.\n\nTransform the user’s text according to these guidelines and return only the revised text without additional commentary."
    },
    {
        "role": "user",
        "content": sent
    }
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

输出：

```md
Explore more nature-themed content online.
```

微软答案：

```md
More nature themes are available online.
```



其他例子：

```md
input: Are you sure you want to delete the file? It might be gone forever.
output: Confirm the file deletion. If you are certain, proceed to delete the file.

input2: Did you remember to install the latest update? It’s important for security.
output2: Install the latest update to ensure security.
```



#### 测试实例2: Use semicolons sparingly. 

提示词：

```python
sent = "Ensure that you have a backup; check your network connectivity; and verify that your firewall is active."

messages = [
    {
        "role": "system",
        "content": "You are an expert writer specializing in Microsoft style. The text you receive may contain semicolons, which are characteristic of a more formal, academic writing style. Your task is to transform the text by replacing semicolons with two separate sentences or a list when possible. For example:\n\nNot Microsoft style:\n\"Antivirus software helps stop unwanted programs from accessing your computer; ensure that your antivirus software is up to date. For more information, see Semicolons (Chapter 9).\"\n\nMicrosoft style:\n\"Antivirus software helps stop unwanted programs from accessing your computer. Make sure your antivirus software is up to date.\"\n\nPlease rewrite the provided text following these guidelines and return only the revised text without any additional commentary."
    },
    {
        "role": "user",
        "content": sent
    }
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```
输出：
```
Ensure that you have a backup. Check your network connectivity. Verify that your firewall is active.
```

#### 小结

通过单个逐一描述修改规则，模型是可以理解需求的，可以将单一的规则变为编辑的按钮，由人工选用。



## 微调模型

如果希望模型能全文改写为微软的风格，则需要通过微调模型的方式，使得模型能主动探测所需要的规则。



### 数据准备

```json
[
  {
    "text": "<|im_start|>system<|im_sep|>You are a Microsoft Style Editor. Your role is to review and edit text to ensure it follows Microsoft's style guidelines: clear, concise, and professional language, with appropriate punctuation and formatting. Always aim for clarity and consistency, transforming technical or dense content into easily understandable text.<|im_end|>\n<|im_start|>user<|im_sep|>Antivirus software helps stop unwanted programs from accessing your computer; ensure that your antivirus software is up to date.<|im_end|>\n<|im_start|>assistant<|im_sep|>Antivirus software helps stop unwanted programs from accessing your computer. Make sure your antivirus software is up to date.<|im_end|>"
  },
  {
    "text": "<|im_start|>system<|im_sep|>You are a Microsoft Style Editor. Your role is to review and edit text to ensure it follows Microsoft's style guidelines: clear, concise, and professional language, with appropriate punctuation and formatting. Always aim for clarity and consistency, transforming technical or dense content into easily understandable text.<|im_end|>\n<|im_start|>user<|im_sep|>Open the settings menu; select 'Privacy'; adjust your preferences.<|im_end|>\n<|im_start|>assistant<|im_sep|>Open the settings menu. Select 'Privacy' and adjust your preferences.<|im_end|>"
  },
  {
    "text": "<|im_start|>system<|im_sep|>You are a Microsoft Style Editor. Your role is to review and edit text to ensure it follows Microsoft's style guidelines: clear, concise, and professional language, with appropriate punctuation and formatting. Always aim for clarity and consistency, transforming technical or dense content into easily understandable text.<|im_end|>\n<|im_start|>user<|im_sep|>Connect your printer to the network; check that the printer is turned on; start printing.<|im_end|>\n<|im_start|>assistant<|im_sep|>Connect your printer to the network. Check that the printer is turned on. Start printing.<|im_end|>"
  },
  ...
]  
```



1. 安装unsloth

   ```
   %%capture
   import os
   if "COLAB_" not in "".join(os.environ.keys()):
       !pip install unsloth
   else:
       # Do this only in Colab and Kaggle notebooks! Otherwise use pip install unsloth
       !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
       !pip install --no-deps cut_cross_entropy unsloth_zoo
       !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
       !pip install --no-deps unsloth
   ```

   

2. 加载Phi‑4并配置参数

   ```python
   from unsloth import FastLanguageModel  # FastVisionModel for LLMs
   import torch
   max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
   load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
   
   # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
   fourbit_models = [
       "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
       "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
       "unsloth/Phi-4",  # Phi-4 2x faster!
       "unsloth/Phi-4-unsloth-bnb-4bit",  # Phi-4 Unsloth Dynamic 4-bit Quant
       "unsloth/gemma-2-9b-bnb-4bit",  # Gemma 2x faster!
       "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # Qwen 2.5 2x faster!
       "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
       "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
       "unsloth/Llama-3.2-3B-bnb-4bit",
       "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
   ]  # More models at https://docs.unsloth.ai/get-started/all-our-models
   
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name = "unsloth/Phi-4",
       max_seq_length = max_seq_length,
       load_in_4bit = load_in_4bit,
       # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
   )
   ```

   

3. 添加LoRA PEFT

   ```python
   model = FastLanguageModel.get_peft_model(
       model,
       r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
       target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
       lora_alpha = 16,
       lora_dropout = 0, # Supports any, but = 0 is optimized
       bias = "none",    # Supports any, but = "none" is optimized
       # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
       use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
       random_state = 3407,
       use_rslora = False,  # We support rank stabilized LoRA
       loftq_config = None, # And LoftQ
   )
   ```

   

4. 添加微调数据

   ```
   from datasets import load_dataset
   
   # 假设文件名为 data.json
   dataset = load_dataset("json", data_files="/content/standardized-ms-style.json", split="train")
   dataset['text'][1]
   ```

   ```md
   <|im_start|>system<|im_sep|>You are a Microsoft Style Editor. Your role is to review and edit text to ensure it follows Microsoft's style guidelines: clear, concise, and professional language, with appropriate punctuation and formatting. Always aim for clarity and consistency, transforming technical or dense content into easily understandable text.<|im_end|>\n<|im_start|>user<|im_sep|>Open the settings menu; select 'Privacy'; adjust your preferences.<|im_end|>\n<|im_start|>assistant<|im_sep|>Open the settings menu. Select 'Privacy' and adjust your preferences.<|im_end|>
   ```

   

5. 微调模型

   ```python
   from trl import SFTTrainer
   from transformers import TrainingArguments, DataCollatorForSeq2Seq
   from unsloth import is_bfloat16_supported
   
   trainer = SFTTrainer(
       model = model,
       tokenizer = tokenizer,
       train_dataset = dataset,
       dataset_text_field = "text",
       max_seq_length = max_seq_length,
       data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
       dataset_num_proc = 2,
       packing = False, # Can make training 5x faster for short sequences.
       args = TrainingArguments(
           per_device_train_batch_size = 2,
           gradient_accumulation_steps = 4,
           warmup_steps = 5,
           # num_train_epochs = 1, # Set this for 1 full training run.
           max_steps = 30,
           learning_rate = 2e-4,
           fp16 = not is_bfloat16_supported(),
           bf16 = is_bfloat16_supported(),
           logging_steps = 1,
           optim = "adamw_8bit",
           weight_decay = 0.01,
           lr_scheduler_type = "linear",
           seed = 3407,
           output_dir = "outputs",
           report_to = "none", # Use this for WandB etc
       ),
   )
   ```

   各参数含义：
   SFTTrainer 构造器参数

   | 参数                 | 含义                                                         |
   | -------------------- | ------------------------------------------------------------ |
   | `model`              | 待微调的模型实例，通常是 `transformers` 的 `PreTrainedModel`（如 `AutoModelForSeq2SeqLM`）。 |
   | `tokenizer`          | 与模型配套的分词器，用于将原始文本转换成模型可接收的 token ID 序列，并负责解码输出。 |
   | `train_dataset`      | 训练用的数据集对象，应实现 PyTorch Dataset 接口，返回样本字典（至少包含 `"text"` 字段）。 |
   | `dataset_text_field` | 指定数据集中，承载文本输入的字段名（本例中为 `"text"`），Trainer 会从此字段获取原始字符串进行编码。 |
   | `max_seq_length`     | 序列最大长度（以 token 数计），对超过该长度的输入进行截断，对不足的输入进行填充（pad）。 |
   | `data_collator`      | 用于组 batch 时的“数据拼接”函数，本例使用 `DataCollatorForSeq2Seq`，可自动填充（pad）并构造 Seq2Seq 任务所需的 decoder 输入。 |
   | `dataset_num_proc`   | 数据预处理（如 tokenization、映射函数）时的多进程数，可加速大规模数据集的预处理。 |
   | `packing`            | 是否启用“packing”技术：将若干短序列拼接到一起填满 `max_seq_length`，对短文本训练速度有显著加速；`False` 则关闭此优化。 |

   TrainingArguments 超参数配置

   | 参数                          | 含义                                                         |
   | ----------------------------- | ------------------------------------------------------------ |
   | `per_device_train_batch_size` | 每个设备（GPU/CPU）上真实执行的微调 batch 大小；结合 `gradient_accumulation_steps` 可模拟更大的全局 batch。 |
   | `gradient_accumulation_steps` | 梯度累积步数：每过此数目步才进行一次参数更新，用于在显存受限情况下获得等同于 `batch_size × accum_steps` 的效果。 |
   | `warmup_steps`                | 学习率预热步数：训练初期将学习率从 0 线性升到设定值，帮助模型更稳定地收敛。 |
   | `max_steps`                   | 最大训练步数：达到此步数后即停止训练（与 `num_train_epochs` 二选一，若同时设置则以先触发者为准）。 |
   | `learning_rate`               | 初始学习率（LR），决定每次参数更新的步长；需根据模型规模、batch 大小等调优。 |
   | `fp16`                        | 是否开启 16-bit 半精度训练（FP16），可显著节省显存并加速；本例在不支持 BF16 时启用。 |
   | `bf16`                        | 是否开启 Brain Floating Point 16 (BF16) 训练，若硬件支持则优先使用 BF16，否则退回 FP16。 |
   | `logging_steps`               | 日志记录间隔（步数）：每训练多少步在控制台输出一次 loss、lr 等指标。 |
   | `optim`                       | 优化器类型，本例用 `"adamw_8bit"` 表示基于 8-bit 量化 AdamW，可进一步降低显存占用。 |
   | `weight_decay`                | 权重衰减系数（L2 正则化），帮助防止过拟合，一般在 0.01 左右。 |
   | `lr_scheduler_type`           | 学习率调度策略，本例用 `"linear"`，即先 warmup 再线性衰减到 0。 |
   | `seed`                        | 全局随机种子，保证可复现（Shuffle、初始化等一致）。          |
   | `output_dir`                  | 模型检查点和训练日志保存目录；训练过程中会在此目录下写入 `checkpoint-xxx`。 |
   | `report_to`                   | 指定日志/指标上报目标，例如 `"wandb"`、`"tensorboard"` 等；设置为 `"none"` 则关闭所有外部上报。 |

6. 仅训练助手输出，忽略用户输入损失

   ```python
   from unsloth.chat_templates import train_on_responses_only
   
   trainer = train_on_responses_only(
       trainer,
       instruction_part="<|im_start|>user<|im_sep|>",
       response_part="<|im_start|>assistant<|im_sep|>",
   )
   ```

   

7. 数据验证

   ```python
   tokenizer.decode(trainer.train_dataset[5]["input_ids"])
   ```

   ```
   <|im_start|>system<|im_sep|>You are a Microsoft Style Editor. Your role is to review and edit text to ensure it follows Microsoft's style guidelines: clear, concise, and professional language, with appropriate punctuation and formatting. Always aim for clarity and consistency, transforming technical or dense content into easily understandable text.<|im_end|><|im_start|>user<|im_sep|>Download the file; install the software; and restart your computer.<|im_end|><|im_start|>assistant<|im_sep|>Download the file. Install the software. Restart your computer.<|im_end|>
   ```

   ```python
   space = tokenizer(" ", add_special_tokens = False).input_ids[0]
   tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
   ```

   ```md
                                                                                  Download the file. Install the software. Restart your computer.<|im_end|>
                                                                                  
                                                                         
   ```

8. 显示内存统计信息

   ```python
   # @title Show current memory stats
   gpu_stats = torch.cuda.get_device_properties(0)
   start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
   max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
   print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
   print(f"{start_gpu_memory} GB of memory reserved.")
   ```

   ```md
   GPU = NVIDIA A100-SXM4-40GB. Max memory = 39.557 GB.
   10.0 GB of memory reserved.
   ```

9. 开始训练

   ```python
   trainer_stats = trainer.train()
   ```

   ```md
   INFO|trainer.py:917] 2025-03-02 05:38:02,976 >> The following columns in the training set don't have a corresponding argument in `PeftModelForCausalLM.forward` and have been ignored: text. If text are not expected by `PeftModelForCausalLM.forward`,  you can safely ignore this message.
   [WARNING|<string>:215] 2025-03-02 05:38:03,057 >> ==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
      \\   /|    Num examples = 521 | Num Epochs = 1
   O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 4
   \        /    Total batch size = 8 | Total steps = 30
    "-____-"     Number of trainable parameters = 65,536,000
    [30/30 01:07, Epoch 0/1]
   Step	Training Loss
   1	2.714500
   2	1.765400
   3	1.759100
   4	1.596600
   5	1.414300
   6	1.047200
   7	1.303400
   8	0.344500
   9	0.708700
   10	0.537400
   11	0.620300
   12	0.443100
   13	1.297000
   14	0.812800
   15	0.524500
   16	0.923000
   17	0.266100
   18	0.695000
   19	0.635100
   20	0.504400
   21	0.346200
   22	0.379400
   23	0.462500
   24	0.253700
   25	0.223400
   26	1.184900
   27	0.726200
   28	0.578300
   29	0.287900
   30	0.174400
   
   ```

   

10. 查看训练后的内容

    ```
    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    ```

    ```md
    76.2308 seconds used for training.
    1.27 minutes used for training.
    Peak reserved memory = 10.959 GB.
    Peak reserved memory for training = 0.959 GB.
    Peak reserved memory % of max memory = 27.704 %.
    Peak reserved memory for training % of max memory = 2.424 %.
    ```

11. 使用微调后的模型推理

    ```python
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    
    messages = [
        {
            "role": "system", "content": "You are a Microsoft Style Editor. Your role is to review and edit text to ensure it follows Microsoft's style guidelines: clear, concise, and professional language, with appropriate punctuation and formatting. Always aim for clarity and consistency, transforming technical or dense content into easily understandable text."
        },
        {
            "role": "user", "content": "A significant amount of people access the website monthly."
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(
        input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
        use_cache = True, temperature = 1.5, min_p = 0.1
    )
    ```

    ```
    A significant number of people access the website monthly.<|im_end|>
    ```

    

12. 保存模型到Hugging Face

    ```python
    # Save to 8bit Q8_0
    if False: model.save_pretrained_gguf("model", tokenizer,)
    # Remember to go to https://huggingface.co/settings/tokens for a token!
    # And change hf to your username!
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")
    
    # Save to 16bit GGUF
    if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    if True: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "hf_MqJZBiJQUEdZYMzIsGqP....M")
    
    # Save to q4_k_m GGUF
    if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")
    
    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "hf/model", # Change hf to your username!
            tokenizer,
            quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
            token = "", # Get a token at https://huggingface.co/settings/tokens
        )
    ```



## 使用微调后的模型

因为模型是量化后的，需要使用支持量化模型的推理框架，这里选择llama.cpp



1. 安装llama-cpp-python

   ```
   !pip install llama-cpp-python
   ```

   

2. 加载模型

   ```python
   from llama_cpp import Llama
   
   llm = Llama.from_pretrained(
   	repo_id="galty687/Microsoft-Style-Editor",
   	filename="unsloth.F16.gguf",
   )
   ```

   

3. 运行

   ```python
   llm.create_chat_completion(
       messages = [
           {
               "role": "system",
               "content": "You are a Microsoft Style Editor. Your role is to review and edit text to ensure it follows Microsoft's style guidelines: clear, concise, and professional language, with appropriate punctuation and formatting. Always aim for clarity and consistency, transforming technical or dense content into easily understandable text."
           },
           {
               "role": "user",
               "content": "Open the settings menu; select 'Privacy'; adjust your preferences."
           }
       ]
   )
   # Extract and print the generated content from the assistant's response:
   generated_content = result['choices'][0]['message']['content']
   print(generated_content)
   ```

   输出：

   ```
   Open the settings menu. Select 'Privacy'. Adjust your preferences.
   ```

   
