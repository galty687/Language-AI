# 大语言模型与语言学

大模型的语言能力有目共睹，但我们应如何科学地评估其语言能力？在设计语言类产品时，尤其是设定性能基线时，通常需要定义一系列评估标准，以确保模型的输出质量满足预期需求，并具备稳定性和一致性。

然而，在缺乏通用指标的情况下，如何衡量模型的语言表现成为一个挑战。这时，语言学的知识便能发挥关键作用，为模型的评估提供理论依据，帮助设定合理的基线，并确保模型在不同应用场景中的有效性。



## 没有语言学，NLP会怎样？

在我们构建用于实际任务的NLP系统时，语言学不再处于核心地位。那种由语言学家精心制定规则的机器翻译引擎的时代已经一去不复返。然而，系统行为的工程设计只是NLP研究或开发/部署生命周期中的一部分，而在NLP中，语言学专长依然发挥着关键作用——换句话说，NLP在某些方面依赖于语言学。

试想一下，如果当前的建模方法（如词嵌入、Transformer、大语言模型）仅凭对语言运作的粗略理解就得以发展，这个领域会变得怎样。许多系统可能看起来相似，但整个领域的面貌将截然不同。我们对语言学分析的关注将极为肤浅（即仅关注那些在特定应用场景传递信息之外的语言系统规律）。即便是那些主要目标并不与语言学分析紧密相关的任务，我们的工具箱也将大大受限。我们从“RELIES”这一助记符的六个方面来探讨NLP研究的内容。事实上，没有语言学的话：

- **资源**：  
  我们将缺乏像词汇表和语料库这样经过精心策划的数据集，这些数据集体现了对语言、方言、体裁和风格差异的充分理解。我们也不会拥有针对语言系统现象的金标准标注，而只有针对应用现象的标注。

- **评价**：  
  我们不仅缺乏针对语言学任务的金标准评价，对于应用任务，我们也缺乏设计有效人工评价、审视自动指标以及描述挑战系统的语言现象（如照应或方言差异）的专长。

- **低资源环境**：  
  我们将更难理解为什么对英语或法语效果不错的方法，对斯瓦希里语或阿拉帕霍语可能行不通。我们也缺乏能够测试神经符号模型中语言归纳偏差以提高准确性或效率的知识。

- **可解释性与解释**：  
  开发和测试关于像大语言模型这样的黑盒系统如何跨领域处理语言的假设将变得更加困难，我们也将缺少描述许多观察到的模式的适当金属语言。

- **语言研究**：  
  诸如句法解析、共指消解以及文本蕴涵/自然语言推理等经典的计算语言学任务在我们的社区中将不存在。一个纯粹致力于商业技术的NLP领域也会对与学术或社区驱动的语言学工作相关的应用（如记录濒危语言）漠不关心。



>[Natural Language Processing RELIES on Linguistics](https://arxiv.org/html/2405.05966v1) 这篇论文探讨了尽管大语言模型能够生成流畅的文本，而不需要专门设计用于捕捉语法或语义一致性的模块，但自然语言处理（NLP）仍然深受语言学的影响。作者提出了“RELIES”这一缩写，代表以下六个方面： **Resources（资源）**、**Evaluation（评价）**、 **Low-resource settings（低资源环境）**、**Interpretability（可解释性）**、 **Explanation（解释）**、 **Study of language（语言研究）**。



## 演示1：衡量文本的连贯程度。



### 素材

```markdown
提示词：写一个北京大学软件与微电子学院的介绍，限3句话。

ChatGPT 4o: 北京大学软件与微电子学院成立于2002年，致力于培养高水平的软件工程和微电子技术人才，融合计算机科学、工程技术与产业实践。学院依托北大的学术优势，注重国际化、交叉学科与产学研合作，为国家和全球科技创新输送优秀人才。学院设有多个研究方向，包括人工智能、软件工程、集成电路设计等，推动前沿技术的发展和应用。

DeepSeek: 北京大学软件与微电子学院是教育部首批国家示范性软件学院之一，聚焦软件工程、微电子、集成电路、人工智能等前沿领域，构建“学科交叉、产学融合”的创新教育体系。学院依托北京大学学科优势，以培养高层次、复合型领军人才为目标，通过产学研协同育人模式，推动核心技术研发与产业转化。其师资团队汇聚国内外顶尖学者，科研实力位居全球前列，为国家信息产业发展和科技自主创新持续输出尖端成果与专业人才。

Qwen: 北京大学软件与微电子学院是中国顶尖的软件与微电子领域高等教育机构，致力于培养具有创新能力和国际视野的高端技术人才。学院以跨学科融合为特色，聚焦软件工程、集成电路设计与应用等前沿领域，推动产学研深度结合。依托北大的学术积淀和资源优势，学院不断探索信息技术发展的新方向，服务于国家重大战略需求。
```



### 工具

[TextDescriptives](https://hlasse.github.io/TextDescriptives/usingthepackage.html) 是一个用于文本分析的 Python 库，旨在计算各种文本特征（textual features），包括基本统计、可读性指标、句法复杂度和文本情感等。它基于 `spaCy` 进行文本处理，并提供了一套易用的 API，适用于 NLP 研究、文本数据探索和自动化文本评估。

### **主要功能**

1. **基本文本统计**：字数、词数、句数、平均单词长度、平均句子长度等。
2. **可读性指标**：Flesch-Kincaid 级别、Gunning Fog 指数、SMOG 指数等。
3. **词汇复杂度**：词汇多样性（TTR、MTLD、VOCD）、长词比例、罕见词比例等。
4. **句法复杂度**：依存关系分析、从句比例、平均从句深度等。
5. **文本情感分析**（可选模块）：结合 NLP 模型进行情感倾向分析。



> **软件安装**：

```bash
# 安装 TextDescriptives 库，用于计算文本的各种统计指标
pip install textdescriptives

# 安装 spaCy，自然语言处理库
pip install spacy

# 下载并安装 spaCy 的中文语言模型 zh_core_web_sm
python -m spacy download zh_core_web_sm

```



### 计算连贯

```python
import spacy
import textdescriptives as td

# 加载中文语言模型
nlp = spacy.load("zh_core_web_md")

# 在处理管道中添加 TextDescriptives 的衔接性组件
nlp.add_pipe("textdescriptives/coherence")

# 定义待分析的中文文本
texts = {
    "chatgpt": "北京大学软件与微电子学院成立于2002年，致力于培养高水平的软件工程和微电子技术人才，融合计算机科学、工程技术与产业实践。学院依托北大的学术优势，注重国际化、交叉学科与产学研合作，为国家和全球科技创新输送优秀人才。学院设有多个研究方向，包括人工智能、软件工程、集成电路设计等，推动前沿技术的发展和应用。",
    "deepseek": "北京大学软件与微电子学院是教育部首批国家示范性软件学院之一，聚焦软件工程、微电子、集成电路、人工智能等前沿领域，构建“学科交叉、产学融合”的创新教育体系。学院依托北京大学学科优势，以培养高层次、复合型领军人才为目标，通过产学研协同育人模式，推动核心技术研发与产业转化。其师资团队汇聚国内外顶尖学者，科研实力位居全球前列，为国家信息产业发展和科技自主创新持续输出尖端成果与专业人才。",
    "qwen": "北京大学软件与微电子学院是中国顶尖的软件与微电子领域高等教育机构，致力于培养具有创新能力和国际视野的高端技术人才。学院以跨学科融合为特色，聚焦软件工程、集成电路设计与应用等前沿领域，推动产学研深度结合。依托北大的学术积淀和资源优势，学院不断探索信息技术发展的新方向，服务于国家重大战略需求。"
}

# 处理文本并提取衔接性指标
for key, text in texts.items():
    doc = nlp(text)
    coherence_df = td.extract_df(doc)
    print(f"文本 '{key}' 的衔接性指标：")
    print(coherence_df)
    print("\n")

```



**计算结果**

| 模型      | text（部分截取）                                                                                                                                  | first_order_coherence | second_order_coherence |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|------------------------|
| chatgpt   | 北京大学软件与微电子学院成立于2002年，致力于培养高水平的软件工程和微电子技术人才，融合计算机科学、工程技术与产业实践。学院依托北大的学术优势，… | 0.874783              | 0.89672                |
| deepseek  | 北京大学软件与微电子学院是教育部首批国家示范性软件学院之一，聚焦软件工程、微电子、集成电路、人工智能等前沿领域，构建“学科交叉、产学融合”的创… | 0.898429              | 0.845563               |
| qwen      | 北京大学软件与微电子学院是中国顶尖的软件与微电子领域高等教育机构，致力于培养具有创新能力和国际视野的高端技术人才。学院以跨学科融合为特色，聚焦软… | 0.842737              | 0.88426                |

> **first order coherence（第一阶衔接性）**
>  该指标主要衡量文本中**相邻句子之间**的衔接和连贯程度。它关注的是句子与句子之间的直接联系，比如是否有实体（如人名、地名、概念等）的连续出现和合理转换，从而保证读者能顺畅地从一个句子过渡到下一个句子。
>
> **second order coherence（第二阶衔接性）**
>  该指标则侧重于**更长距离或整体文本结构**的连贯性。它不仅考虑相邻句子的关系，还关注文本在更大范围内的组织和主题发展，衡量文本整体的结构性连贯，比如主题是否前后一致、论述是否有层次性和逻辑性。
>
> 简单来说，第一阶衔接性关注局部句子之间的连接，而第二阶衔接性反映的是文本全局的连贯性和结构完整性。



### 连贯性的语言学解释

在语言学中，**连贯性（coherence）** 是使文本或话语在整体上具有意义和逻辑性的特性。连贯性确保信息能够自然流畅地传达，使读者或听者能够理解文本的逻辑。

在语言交际中，连贯性决定了文本是否具有可读性和可理解性。一个连贯的文本不仅句子通顺，而且整体信息的组织方式合理，使人能够顺利理解其中的思想。

### 为什么连贯性重要？

连贯性是**有效沟通的核心要素**。无论是在日常交流、学术写作还是商业文本中，连贯性都能提高文本的可理解性，确保信息传递的清晰性。以下是连贯性的几个重要作用：

- **提高文本的易读性**：使句子之间逻辑通顺，减少理解障碍。
- **增强信息的连贯性**：帮助读者建立逻辑联系，使文本更具整体感。
- **提升语言表达能力**：在写作和口语表达中，连贯性使信息传递更具逻辑性和说服力。

### 连贯性 vs. 衔接性

**连贯性（coherence）** 和 **衔接性（cohesion）** 是文本语言学中两个不同但相关的概念。

| **概念**                | **定义**                                           | **特点**                             |
| ----------------------- | -------------------------------------------------- | ------------------------------------ |
| **连贯性（coherence）** | 强调文本的整体逻辑性，使信息自然流畅地传达         | 依赖上下文、逻辑关系、读者的背景知识 |
| **衔接性（cohesion）**  | 通过语言手段（如代词、连接词等）建立句子之间的联系 | 依赖具体的词汇和语法结构             |

**示例：**

1. **衔接性强但缺乏连贯性**：
   - “猫在垫子上。它是黑色的。垫子很舒服。”
   - 这些句子之间存在衔接（代词“它”和重复的“垫子”），但整体意义可能不够清晰。
2. **连贯性强但缺乏衔接性**：
   - “A: 天气怎么样？”
   - “B: 下雨了。”
   - 这里没有明显的衔接词，但语境使对话连贯。

### 连贯性的认知机制

连贯性不仅仅依赖于文本内部的语言结构，还涉及**读者或听者的认知能力**，即他们如何利用已有知识来理解文本。这与**图式理论（schema theory）** 密切相关。

**图式理论（Schema Theory）** 由 F. C. Bartlett 提出，认为人们通过已有的**知识框架（schema）** 来组织和理解新信息。

**示例：**

- **专家 vs. 初学者的理解差异**
  - 一篇关于**人工智能（AI）** 的论文，对 AI 研究员而言可能是连贯的，但对普通人来说可能难以理解。
  - 这表明连贯性**取决于受众的背景知识**。

### 如何实现连贯性

#### 主题-述题结构（Theme-Rheme Structure）

**主题（theme）** 是句子的出发点，通常是已知信息，而**述题（rheme）** 提供新信息。

**示例：**

1. 主题-述题连贯性示例：
   - **句子 1**：“彼得最近去伦敦看奥运会。”（主题：彼得；述题：去伦敦看奥运会）
   - **句子 2**：“他对开幕式印象深刻。”（主题：他；述题：对开幕式印象深刻）
   - 主题“彼得”在第二句话中被代词“他”承接，保证了连贯性。

#### 逻辑关系与连贯性

以下是几种常见的逻辑关系，它们在保持文本连贯性方面发挥着重要作用：

- **因果关系**（例如：“因为…所以…”）
- **对比关系**（例如：“然而”、“但是”）
- **递进关系**（例如：“此外”、“更进一步”）

**示例：**

- **无逻辑关系**：“我喜欢苹果。今天下雨了。”（不连贯）
- **有逻辑关系**：“我喜欢苹果，因为它们很甜。”（连贯）

#### 语境与连贯性

连贯性不仅取决于文本内部的结构，还受到外部**语境（context）** 的影响。例如：

- **对话中的连贯性**
  - A: “你吃过饭了吗？”
  - B: “刚吃过。”
  - 这里没有显性衔接手段，但在语境下是连贯的。

### 连贯性的层级

#### 局部连贯与整体连贯

**连贯性可以分为局部连贯（local coherence）和整体连贯（global coherence）。**

| **层级**     | **定义**             | **示例**                                         |
| ------------ | -------------------- | ------------------------------------------------ |
| **局部连贯** | 句子之间的逻辑联系   | “我去商店了。我买了一瓶牛奶。”（合理的事件顺序） |
| **整体连贯** | 文本整体的结构和意义 | 一篇文章具有清晰的引言、主体和结论               |

### 连贯性的实际应用

#### 在写作中的应用

在学术写作、商业文案和新闻报道中，连贯性都起着关键作用。

**写作技巧：**

- **使用逻辑连接词**（如“因此”、“然而”）
- **保持主题一致**（如一个段落围绕一个核心思想）
- **确保信息顺序合理**（如按时间顺序或因果关系排列）

#### 在口语交流中的应用

**连贯的表达** 在日常对话、演讲和教学中至关重要。

**示例：**

- **不连贯的表达：** “我今天去商场。你知道数学很难学。”
- **连贯的表达：** “我今天去商场买了一本数学书，因为数学很难学。”

### 结论

连贯性是语言理解和表达的核心特征，它依赖于**逻辑关系、语境、主题-述题结构以及受众的背景知识**。无论是写作、口语表达还是学术研究，连贯性都是确保有效沟通的重要因素。

**思考与练习：**

1. 请找一个新闻报道，分析它是如何实现连贯性的。
2. 重新组织以下句子，使其更加连贯：
   - “他喜欢跑步。他昨天去超市买了一瓶水。”
   - “她学习很努力。她最近在研究人工智能。”



## 衔接的语言学概念

**衔接（Cohesion）** 是语言学中用于描述**文本内部各部分之间连接方式**的概念。它确保句子和段落之间形成有机整体，使读者或听者能够顺畅理解文本。  英国语言学家 **M.A.K. Halliday 和 Ruqaiya Hasan** 在 1976 年的著作 *Cohesion in English* 中提出了这一概念，认为衔接是**语篇（discourse）**连贯性的关键因素之一。

衔接是**表层语言结构**的显性特征，它通过具体的语言手段（如代词、连接词、重复词汇等）建立句子之间的逻辑关系，使文本具有整体性。衔接的有效运用可以提高文本的可读性，并减少冗余表达。

### 语篇衔接的主要类型
Halliday 和 Hasan 提出了五种主要的衔接手段：指代、替代、省略、连接和词汇衔接。以下分别介绍其定义、功能及具体示例。

#### 指代（Reference）
**定义**：指代是一种语言现象，指的是用**代词、指示词**等语言单位指代前文或后文中的实体，以避免重复。  
**功能**：
- 使信息紧密衔接，减少冗余。
- 提供语境线索，帮助读者理解句间关系。

**示例**：
- **我昨天见到了张三。他**看起来很高兴。（“他”指代“张三”）
- **这本书**很好，我想借一下。（“这”指代一本特定的书）

指代可以分为：
1. **内指（Endophoric reference）**：指代文本内的信息。
2. **外指（Exophoric reference）**：指代文本外的现实世界实体。

#### 替代（Substitution）
**定义**：替代是指用**另一个词或短语**代替已经出现的内容，以避免重复表达。  
**功能**：
- 通过变换表达方式，使语言更加丰富。
- 使句子结构更加紧凑。

**示例**：
- A: 你想喝咖啡吗？  
- B: **我想喝茶。**（这里用“茶”替代了可能会重复的“咖啡”）

- A: 我喜欢这部电影。  
- B: **我也是。**（“也是”替代了完整的“我也喜欢这部电影”）

#### 省略（Ellipsis）
**定义**：省略是指在语境允许的情况下，**省略已知信息**，使表达更加简洁。  
**功能**：

- 避免冗长表达，提高语言效率。
- 让读者或听者通过上下文推断缺失的内容。

**示例**：

- A: 你吃饭了吗？  
- B: **吃了。**（省略“我吃了饭”）

- A: 你喜欢哪部电影？  
- B: **这部。**（省略“我喜欢这部电影”）

#### 连接（Conjunction）
**定义**：连接是指通过**连接词或逻辑标记词**表明句子之间的关系，如因果、转折、递进等。  
**功能**：
- 使文本逻辑清晰，方便读者理解句间关系。
- 提供层次感，使信息更加条理化。

**示例**：
- **因为**天气不好，我们取消了旅行。（因果关系）
- 我喜欢喝咖啡，**但是**不喜欢喝茶。（转折关系）
- 先完成作业，**然后**去玩游戏。（时间顺序）

常见的连接关系如下：

| 连接关系 | 连接词示例 |
|---------|----------|
| **因果** | 因为、因此、所以 |
| **转折** | 但是、然而、不过 |
| **递进** | 而且、此外、并且 |
| **并列** | 和、以及、同时 |
| **时间** | 然后、随后、接着 |

#### 词汇衔接（Lexical Cohesion）
**定义**：词汇衔接指的是通过**词汇的复现、同义词、近义词、反义词等**手段，使文本保持连贯性。  
**功能**：

- 维持语篇的主题一致性。
- 通过不同表达方式丰富语篇的词汇多样性。

**示例**：
- **复现（Repetition）：**  
  - 昨天我看了一场**足球比赛**。这场**比赛**非常精彩。（“比赛”重复出现）
- **同义词（Synonymy）：**  
  - 我读了一本**小说**。这本**书**让我很感动。（“小说”和“书”是近义表达）
- **反义词（Antonymy）：**  
  - 白天很热，**晚上**却很冷。（“白天”和“晚上”形成对比）

### 衔接与连贯的区别
**衔接（Cohesion）** 和 **连贯（Coherence）** 虽然都与语篇的组织有关，但它们是不同的概念：

| 术语 | 定义 | 例子 |
|------|------|------|
| **衔接（Cohesion）** | 通过显性语言手段（代词、连接词、重复词汇等）保持句子之间的连接 | “今天很冷，所以我穿了一件厚外套。”（使用连接词“所以”衔接前后句子） |
| **连贯（Coherence）** | 语篇在语义上的合理性和可理解性，强调逻辑关系 | “天气很好，我们决定去野餐。”（没有明显的衔接手段，但逻辑清晰，符合语义连贯性） |

**示例对比**：
- **衔接好但不连贯**：  
  - “我今天吃了苹果。苹果是水果。水果很健康。”  
  - 这里的“苹果”和“水果”形成词汇衔接，但句子之间缺乏清晰的逻辑连贯性。
  
- **连贯但衔接性低**：  
  - “天气很好，我们决定去野餐。”  
  - 这里没有显式的衔接手段，但语义逻辑清晰，因此是连贯的。

### 语篇衔接的作用
良好的衔接有助于：
1. **提高文本的可读性**：增强信息的连贯性，使读者能够流畅地理解内容。
2. **减少冗余表达**：通过指代、替代和省略等手段，提高语言表达的简洁性。
3. **保证信息的逻辑连贯性**：确保文本内容有条理，逻辑清晰。

### 小结
本章介绍了语篇衔接的基本概念，并分析了五种主要的衔接手段：指代、替代、省略、连接和词汇衔接。  
衔接是语篇结构的表层特征，它通过显性语言手段增强文本的整体性，而连贯性则涉及语篇的语义合理性和逻辑一致性。掌握衔接手段能够提高文本的表达质量，使信息更加连贯、清晰。

练习：

请将下列句子重新组织，使其衔接更加自然流畅：

**原句1：**

- 她买了一本新书。她在地铁上读这本书。
- 这本书讲的是人工智能。

**原句2：**

- 会议将在明天举行。会议的主题是大模型。
- 许多专家将参加这次会议。
- 会议结束后，大家会进行自由讨论。



---

## 演示2：衡量文本的可读性。

```bash
#下载英文语言模型

python -m spacy download en_core_web_sm
```



### 素材

用Chatgpt 4o写了三段关于 Readability的介绍。

```markdown
# Instruction Prompt

"Write an introduction about 'Readability' in three different versions, each tailored to a specific audience:

    1. Pupils (elementary school students) – Use simple words and short sentences to ensure young children can easily understand.
    2. Middle school students – Use slightly more advanced vocabulary and sentence structures appropriate for teenagers.
    3. College students – Use formal and academic language suitable for higher education, including technical terms if necessary."


# Results

## Pupil Version (Elementary School Students)
What is Readability?
Readability means how easy or hard it is to read something. If a book or story has simple words and short sentences, it is easy to read. But if it has big words and long sentences, it might be harder to understand.

Imagine you are reading a fairy tale. The words are easy, and you can enjoy the story. But if you try to read a book for adults, the words may be too difficult. Readability helps writers choose the right words so everyone can understand and enjoy reading!


## Middle School Version
Readability is how easy or difficult a piece of writing is to understand. It depends on things like word choice, sentence length, and how information is organized.

For example, a children's book is very readable because it uses short words and simple sentences. On the other hand, a science textbook may be harder to read because it has long sentences and difficult words. Writers and teachers use readability to make sure books match the reader's level, so learning and reading can be more enjoyable!

## College Student Version

Introduction to Readability
Readability refers to the ease with which a reader can understand a written text. It is influenced by factors such as vocabulary complexity, sentence structure, and text organization. Readability is often measured using formulas such as the Flesch-Kincaid Readability Test, which evaluates sentence length and word difficulty to determine reading level.

Ensuring appropriate readability is crucial in education, journalism, and technical writing. Writers must consider their audience when crafting content, as overly complex language can hinder comprehension, while overly simplified text may fail to convey necessary depth. Readability plays a key role in effective communication, helping to bridge the gap between information and understanding.
```



### 计算可读性

```python
import spacy
import textdescriptives as td

# Load the spaCy model and add the textdescriptives readability pipe
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textdescriptives/readability")

# Define the three text samples
texts = {
    "Pupil Version": (
        "What is Readability? "
        "Readability means how easy or hard it is to read something. "
        "If a book or story has simple words and short sentences, it is easy to read. "
        "But if it has big words and long sentences, it might be harder to understand. "
        "Imagine you are reading a fairy tale. The words are easy, and you can enjoy the story. "
        "But if you try to read a book for adults, the words may be too difficult. "
        "Readability helps writers choose the right words so everyone can understand and enjoy reading!"
    ),
    "Middle School Version": (
        "Readability is how easy or difficult a piece of writing is to understand. "
        "It depends on things like word choice, sentence length, and how information is organized. "
        "For example, a children's book is very readable because it uses short words and simple sentences. "
        "On the other hand, a science textbook may be harder to read because it has long sentences and difficult words. "
        "Writers and teachers use readability to make sure books match the reader's level, so learning and reading can be more enjoyable!"
    ),
    "College Student Version": (
        "Introduction to Readability "
        "Readability refers to the ease with which a reader can understand a written text. "
        "It is influenced by factors such as vocabulary complexity, sentence structure, and text organization. "
        "Readability is often measured using formulas such as the Flesch-Kincaid Readability Test, which evaluates sentence length and word difficulty to determine reading level. "
        "Ensuring appropriate readability is crucial in education, journalism, and technical writing. "
        "Writers must consider their audience when crafting content, as overly complex language can hinder comprehension, while overly simplified text may fail to convey necessary depth. "
        "Readability plays a key role in effective communication, helping to bridge the gap between information and understanding."
    )
}

# Process each text and print its readability scores
results = {}
for level, text in texts.items():
    doc = nlp(text)
    # Retrieve the readability metrics as a dictionary
    readability = doc._.readability
    # Extract to a DataFrame (this returns a pandas DataFrame)
    df = td.extract_df(doc)
    results[level] = (readability, df)
    print(f"\n=== {level} ===")
    print("Readability Dictionary:")
    print(readability)
    print("\nReadability DataFrame:")
    print(df)

# Optionally, if you wish to combine results for comparison:
import pandas as pd

dfs = []
for level, (readability, df) in results.items():
    df['Version'] = level
    dfs.append(df)
combined_df = pd.concat(dfs, ignore_index=True)
print("\nCombined Readability Results:")
print(combined_df)

```



输出结果

| text                                                         | token_length_mean | token_length_median | token_length_std | sentence_length_mean | sentence_length_median | sentence_length_std | syllables_per_token_mean | syllables_per_token_median | syllables_per_token_std | n_tokens | n_unique_tokens | proportion_unique_tokens | n_characters | n_sentences | flesch_reading_ease | flesch_kincaid_grade | smog | gunning_fog | automated_readability_index | coleman_liau_index |  lix |  rix | Version                 | sample_text                                                  |
| :----------------------------------------------------------- | ----------------: | ------------------: | ---------------: | -------------------: | ---------------------: | ------------------: | -----------------------: | -------------------------: | ----------------------: | -------: | --------------: | -----------------------: | -----------: | ----------: | ------------------: | -------------------: | ---: | ----------: | --------------------------: | -----------------: | ---: | ---: | :---------------------- | :----------------------------------------------------------- |
| What is Readability? Readability means how easy or hard it... |               4.8 |                   4 |          3.24962 |                    5 |                      5 |                   2 |                      1.6 |                          1 |                     1.2 |       10 |               9 |                      0.9 |           49 |           2 |                66.4 |                 5.24 |  nan |          10 |                       3.678 |              6.504 |   25 |    1 | Pupil Version           | What is Readability? Readability means how easy or hard it   |
| Readability is how easy or difficult a piece of writing...   |               4.6 |                 3.5 |              3.2 |                   10 |                     10 |                   0 |                      1.6 |                          1 |                  1.0198 |       10 |              10 |                        1 |           46 |           1 |              61.325 |                 7.19 |  nan |          12 |                       5.236 |              8.288 |   40 |    3 | Middle School Version   | Readability is how easy or difficult a piece of writing      |
| Introduction to Readability Readability refers to the ease with which... |                 6 |                 4.5 |          3.68782 |                   10 |                     10 |                   0 |                      1.9 |                          1 |                 1.37477 |       10 |               8 |                      0.8 |           60 |           1 |              35.945 |                10.73 |  nan |          16 |                       11.83 |              16.52 |   40 |    3 | College Student Version | Introduction to Readability Readability refers to the ease with which |



### 指标解读

#### Gunning-Fog 指数

**来源**：[`Gunning-Fog`](https://en.wikipedia.org/wiki/Gunning_fog_index)

**说明**：Gunning-Fog 指数最初为英语写作设计，但可适用于任何语言。该指数估计了读者在首次阅读时所需具备的正规教育年数。

- 公式：年级水平 = 0.4 × (ASL + PHW)
  - **ASL**：平均句子长度（总词数除以总句数）
  - **PHW**：难词百分比（指具有三个或更多音节的单词的比例）
     **注意**：需要连字符处理以正确划分单词音节。

**含义解读**：
 分数越高，说明文本需要更高的教育水平才能理解，即文本阅读难度越大；分数较低则表示文本较容易理解，适合较低年级的读者。

#### SMOG 指数

**来源**：[`SMOG`](https://en.wikipedia.org/wiki/SMOG)（简单胡言乱语测量）

**说明**：SMOG 指数通过统计文本中多音节单词的数量来估计理解文本所需的教育年数。

- **公式**：
   SMOG 指数 = 1.043 × √(30 × (难词数 / 句子数)) + 3.1291
   **注意**：需要连字符处理。

#### Flesch 阅读易度

**来源**：[`Flesch reading ease`](https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests#Flesch_reading_ease)

**说明**：
 Flesch 阅读易度分数衡量文本的易读性。

- 公式：Flesch 阅读易度 = 206.835 - (1.015 × ASL) - (84.6 × ASW)

  - **ASL**：平均句子长度
- **ASW**：每个单词的平均音节数
  - **含义解读**：
   分数越高，表示文本越容易阅读；分数越低，则说明文本阅读起来更困难，适合受过较高教育水平的读者。

**注意**：需要连字符处理。

#### Flesch-Kincaid 年级水平

**来源**：[`Flesch-Kincaid grade`](https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests#Flesch–Kincaid_grade_level)

**说明**：
 该指标估计理解文本所需的年级水平，主要基于句子长度和单词的音节数。

- **公式**：Flesch-Kincaid 年级 = 0.39 × ASL + 11.8 × ASW - 15.59
- **含义解读**：
   分数表示需要达到相应年级的读者才能较好地理解文本。分数越高，说明文本更复杂；分数较低，则表示文本较为简单易懂。

**注意**：需要连字符处理。

#### 自动可读性指数（ARI）

**来源**：[`Automated readability index`](https://en.wikipedia.org/wiki/Automated_readability_index)

**说明**：
 ARI 是一种自动化可读性测试，基于每个单词的平均字符数和平均句子长度，估计理解文本所需的美国年级水平。

- **公式**：ARI = 4.71 × (字符数 / 词数) + 0.5 × (词数 / 句子数) - 21.43
- **含义解读**：
   分数表示理解文本所需的最低美国学年。分数越高，表示文本结构更复杂，要求读者具备更高的阅读能力；分数较低则说明文本更简单易懂。



#### Coleman-Liau 指数

**来源**：[`Coleman-Liau index`](https://en.wikipedia.org/wiki/Coleman–Liau_index)

**说明**：
 该指数用于估计理解文本所需的美国年级水平，其依据是每 100 个词中的平均字母数和句子数。

- 原始公式：CLI = 0.0588 × L - 0.296 × S - 15.8

  - **L**：每 100 个词的平均字符数
  - **S**：每 100 个词的平均句子数
     在实际实现中，我们对整个文本进行平均，而不仅局限于 100 个词。
  - **含义解读**：
     分数越高，表明理解该文本所需的年级越高，文本阅读难度也越大；分数较低则表示文本较容易理解。

#### Lix 指数

**来源**：[`Lix`](https://en.wikipedia.org/wiki/Lix_(readability_test))（Lesbarhetsindex）

**说明**：
 Lix 指数根据平均句子长度和长词（超过六个字符）的比例来评估文本的可读性。

- **公式**：
   Lix = (词数 / 句子数) + (长词数 × 100 / 词数)
- **含义解读**：
   分数越高，表示文本包含更多长句和长词，难度较大；分数较低则说明文本较为简洁、易于阅读。

#### Rix 指数

**来源**：[`Rix`](https://www.jstor.org/stable/40031755)

**说明**：
 Rix 指数通过计算文本中长词（超过六个字符）在句子中的比例来估计文本的难度。

- **公式**：Rix = 长词数 / 句子数
- **含义解读**：
   分数越高，说明文本中长词比例较大，理解起来较为困难；分数较低则表明文本更为通俗易懂。
