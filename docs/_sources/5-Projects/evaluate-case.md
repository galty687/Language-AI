# 评价模型翻译能力

**EleutherAI 的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**：其评估基准的示例数量中位数为 1,000，平均数为 2,159。组织者建议评估集的最小规模为 300 个示例，但更倾向于至少包含 1,000 个示例，特别是当这些示例是合成时。

 **lm-evaluation-harness** 中包含了翻译任务测评的子任务，本文将用该工具对拟采用的模型进行评估。

1. 安装

   ```bash
   !pip install git+https://github.com/EleutherAI/lm-evaluation-harness
   ```



2. 查看支持的测评任务

   ```
   !lm-eval --tasks list
   ```

   输出（只复制了一行）：

    ```bash
   |bigbench_salient_translation_error_detection_generate_until   
    ```

   > **[Salient Translation Error Detection](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/salient_translation_error_detection)** 是 BIG-bench 基准测试套件中的一个任务，旨在评估语言模型检测翻译中显著错误的能力。该任务的目标是确定模型是否能够识别翻译文本中的关键错误，确保译文的准确性和一致性。目前这个测试集只支持德文和英文的翻译能力测评。
   >
   > **任务特点：**
   >
   > - **错误类型**：模型需要检测翻译中的各种错误类型，包括但不限于：数值错误、命名实体错误、否定或反义词使用错误、修饰语或形容词使用错误、内容遗漏以及事实性错误。
   > - **评估指标**：主要使用准确率（accuracy）等指标来衡量模型在检测翻译错误方面的性能。

   其他详细任务见文末。

   

3. 评估meta的mbart的德英翻译能力

   ```bash
   !lm-eval --model hf \
       --model_args pretrained=facebook/mbart-large-50,trust_remote_code=True \
       --tasks bigbench_salient_translation_error_detection_generate_until \
       --device cuda:0 \
       --batch_size 8 \
       --output_path /content/lm_eval_results3
   
   ```

   输出：

   | Tasks                                                       | Version | Filter | n-shot | Metric      |      |  Value |      | Stderr |
   | ----------------------------------------------------------- | ------: | ------ | -----: | ----------- | ---- | -----: | ---- | -----: |
   | bigbench_salient_translation_error_detection_generate_until |     1.0 | none   |      0 | exact_match | ↑    | 0.0000 | ±    | 0.0000 |

4. 对照GPT-4o的德英翻译能力

   ```
   !pip install tiktoken
   
   import os
   os.environ['OPENAI_API_KEY'] = 'your-api-key'
   
   ```

   

   ```
   !lm_eval --model openai-chatcompletions \
           --model_args model=gpt-4o \
           --tasks bigbench_salient_translation_error_detection_generate_until \
           --device cuda:0 \
           --batch_size 8 \
           --output_path /content/lm_eval_results4
   
   ```

   输出结果：
   
   openai-chat-completions (model=gpt-4o), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 1
   
   | Tasks                                                       | Version | Filter | n-shot | Metric      |      |  Value |      | Stderr |
   | ----------------------------------------------------------- | ------: | ------ | -----: | ----------- | ---- | -----: | ---- | -----: |
   | bigbench_salient_translation_error_detection_generate_until |       1 | none   |      0 | exact_match | ↑    | 0.0341 | ±    | 0.0057 |





## 结果解读

一开始我以为只是Meta的模型能力一般，然后就尝试了使用gpt-4o，其正确率依然低得吓人，于是我仔细看了一下分析结果。



### 数据集

德文句子的来源如下：一组描述（段落）从 [WIT](https://github.com/google-research-datasets/wit) 数据集中抽样获得。利用句子分割器（Stanza）提取了描述的第一句，再请商业翻译服务将句子翻译成英文。随后，将这些句对（原文、译文）提供给人工按照如下要求加工数据：在翻译中引入六类错误中的一种，同时尽量不影响翻译的流畅性。



示例数据：

```json
{
        "input": "Source: Das Einköpfige Ferkelkraut ist eine Pflanzenart aus der Gattung der Ferkelkräuter innerhalb der Familie der Korbblütler.\nTranslation: The two-headed piglet herb is a species of herb in the family.\nThe translation contains an error pertaining to ",
        "target_scores": {
         "Modifiers or Adjectives": 0,
         "Numerical Values": 1,
         "Negation or Antonyms": 0,
         "Named Entities": 0,
         "Dropped Content": 0,
         "Facts": 0
        }
```



### 使用 gpt-o1 实验：

测试集标注的是："Numerical Values": 1,

提示词：

```md
"input": "Source: Der Müggenburger Zollhafen ist ein Hafenbecken in Hamburg-Veddel.\nTranslation: The Müggenburg Harbour is a dock in Hamburg-Veddel, Germany.\nThe translation contains an error pertaining to ",
        "target_scores": {
         "Modifiers or Adjectives": 0,
         "Numerical Values": 0,
         "Negation or Antonyms": 0,
         "Named Entities": 0,
         "Dropped Content": 0,
         "Facts": 0

Please try to find the error in the input and increase the counter in its corresponding error category.
```

输出：

```md
Analysis of the source and translation:

    Source (German):
    Der Müggenburger Zollhafen ist ein Hafenbecken in Hamburg-Veddel.
    Literally: “The Müggenburger Zollhafen is a harbor basin in Hamburg-Veddel.”

    Translation (English):
    The Müggenburg Harbour is a dock in Hamburg-Veddel, Germany.

The term “Müggenburger Zollhafen” is a specific named entity. In the translation, it was changed to “Müggenburg Harbour,” and the word “Zoll” (“customs”) was dropped entirely. Because “Müggenburger Zollhafen” is the proper name for that port facility, losing “Zoll” is effectively an error in translating or preserving the named entity.

Relevant error category: “Named Entities”

Since the specific name (Müggenburger Zollhafen) was altered and partially dropped (“Zollhafen” → “Harbour”), this counts as a mistake in handling a named entity. Therefore, you would increment the counter for Named Entities by 1.
```

使用ChatGPT 多次查证后发现：原译文将“Hafenbecken”错误地译为“dock”，而“Hafenbecken”指的是“港口水域”或“港口盆地”，而不是“码头”。因此，o3-mini认为这一错误属于事实错误，o1认为是一个实体识别的错误。但是答案认为是一个数值错误。

总之，人工标记的逻辑和模型的逻辑不匹配，该数据集作者也未曾通过其他方式证实该数据集的可靠性。这也是小型数据集的通用问题，人工构建的数据集随意性较大，后续大家使用数据集时要注意选择权威可靠的来源。



---



### 其他评估任务

`lm-eval`预制了非常多的任务，具体可以通过下方指令查询

```bash
!lm-eval --tasks list
```



输出的部分结果如下，

| Task                                                    | Path                                                         | Type            |
| ------------------------------------------------------- | ------------------------------------------------------------ | --------------- |
| bigbench_mathematical_induction_multiple_choice         | lm_eval/tasks/bigbench/multiple_choice/mathematical_induction.yaml | multiple_choice |
| bigbench_matrixshapes_generate_until                    | lm_eval/tasks/bigbench/generate_until/matrixshapes.yaml      | generate_until  |
| bigbench_metaphor_boolean_generate_until                | lm_eval/tasks/bigbench/generate_until/metaphor_boolean.yaml  | generate_until  |
| bigbench_metaphor_boolean_multiple_choice               | lm_eval/tasks/bigbench/multiple_choice/metaphor_boolean.yaml | multiple_choice |
| bigbench_metaphor_understanding_generate_until          | lm_eval/tasks/bigbench/generate_until/metaphor_understanding.yaml | generate_until  |
| bigbench_metaphor_understanding_multiple_choice         | lm_eval/tasks/bigbench/multiple_choice/metaphor_understanding.yaml | multiple_choice |
| bigbench_minute_mysteries_qa_generate_until             | lm_eval/tasks/bigbench/generate_until/minute_mysteries_qa.yaml | generate_until  |
| bigbench_misconceptions_generate_until                  | lm_eval/tasks/bigbench/generate_until/misconceptions.yaml    | generate_until  |
| bigbench_misconceptions_multiple_choice                 | lm_eval/tasks/bigbench/multiple_choice/misconceptions.yaml   | multiple_choice |
| bigbench_misconceptions_russian_generate_until          | lm_eval/tasks/bigbench/generate_until/misconceptions_russian.yaml | generate_until  |
| bigbench_misconceptions_russian_multiple_choice         | lm_eval/tasks/bigbench/multiple_choice/misconceptions_russian.yaml | multiple_choice |
| bigbench_mnist_ascii_generate_until                     | lm_eval/tasks/bigbench/generate_until/mnist_ascii.yaml       | generate_until  |
| bigbench_mnist_ascii_multiple_choice                    | lm_eval/tasks/bigbench/multiple_choice/mnist_ascii.yaml      | multiple_choice |
| bigbench_modified_arithmetic_generate_until             | lm_eval/tasks/bigbench/generate_until/modified_arithmetic.yaml | generate_until  |
| bigbench_moral_permissibility_generate_until            | lm_eval/tasks/bigbench/generate_until/moral_permissibility.yaml | generate_until  |
| bigbench_moral_permissibility_multiple_choice           | lm_eval/tasks/bigbench/multiple_choice/moral_permissibility.yaml | multiple_choice |
| bigbench_movie_dialog_same_or_different_generate_until  | lm_eval/tasks/bigbench/generate_until/movie_dialog_same_or_different.yaml | generate_until  |
| bigbench_movie_dialog_same_or_different_multiple_choice | lm_eval/tasks/bigbench/multiple_choice/movie_dialog_same_or_different.yaml | multiple_choice |
| bigbench_movie_recommendation_generate_until            | lm_eval/tasks/bigbench/generate_until/movie_recommendation.yaml | generate_until  |
| bigbench_movie_recommendation_multiple_choice           | lm_eval/tasks/bigbench/multiple_choice/movie_recommendation.yaml | multiple_choice |
| bigbench_mult_data_wrangling_generate_until             | lm_eval/tasks/bigbench/generate_until/mult_data_wrangling.yaml | generate_until  |
| bigbench_multiemo_generate_until                        | lm_eval/tasks/bigbench/generate_until/multiemo.yaml          | generate_until  |
| bigbench_multiemo_multiple_choice                       | lm_eval/tasks/bigbench/multiple_choice/multiemo.yaml         | multiple_choice |
| bigbench_natural_instructions_generate_until            | lm_eval/tasks/bigbench/generate_until/natural_instructions.yaml | generate_until  |

其他任务的部分介绍如下：具体的任务介绍可见：[Tasks](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md)

| 任务系列                    | 描述                                                         | 语言                     |
| --------------------------- | ------------------------------------------------------------ | ------------------------ |
| aclue                       | 专注于古代汉语理解及文化方面的任务。                         | 古汉语                   |
| aexams                      | 与各种学科考试相关的阿拉伯语任务。                           | 阿拉伯语                 |
| agieval                     | 涉及历史数据或与历史及历史文本相关问题的任务。               | 英语，中文               |
| anli                        | 对抗性自然语言推理任务，用以测试模型鲁棒性。                 | 英语                     |
| arabic_leaderboard_complete | 开放阿拉伯语大语言模型排行榜中任务的完整版，侧重于评估反映阿拉伯语理解、领会、文化和遗产特点的模型。注意，其中一些任务是机器翻译的。 | 阿拉伯语（部分机器翻译） |
| arabic_leaderboard_light    | 开放阿拉伯语大语言模型排行榜中任务的简化版（即原基准测试集的10%样本），侧重于评估反映阿拉伯语理解、领会、文化和遗产特点的模型。注意，其中一些任务是机器翻译的。 | 阿拉伯语（部分机器翻译） |

