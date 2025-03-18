# 基于Langchain的提示词开发实践

## LangChain简介

LangChain 是一个开源框架，旨在简化和加速基于大型语言模型 (LLM) 的应用程序开发。它提供了一整套工具和模块，使开发者可以轻松地将语言模型与外部数据源、API、甚至自主决策机制相结合，从而实现数据增强和智能化功能。LangChain 同时支持 Python 和 TypeScript 版本，让不同技术栈的开发者都能利用这一框架快速构建高效、灵活的 AI 应用。其核心理念在于不仅仅将语言模型作为简单的 API 接口使用，而是通过丰富的功能扩展，打造出更加智能和具有自主决策能力的应用程序。



### 安装与准备

如果是在本地运行，建议用虚拟环境。



```
!pip install langchain langchain-openai
```



```python
from langchain_openai.chat_models import ChatOpenAI
from google.colab import userdata
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Instantiate ChatOpenAI with the desired settings
chat = ChatOpenAI(
    temperature=0.6,
    model="gpt-4o",  # Correctly specify the model name (ensure the model name is valid)
    api_key=userdata.get('OPENAI_API_KEY')
)

# Define the conversation messages
messages = [
    SystemMessage(content="Act as a senior software engineer at a startup company."),
    HumanMessage(content="Please can you provide a funny joke about software engineers?")
]

# Invoke the chat model using the provided messages
response = chat.invoke(input=messages)

# Print the response content
print(response.content)

```

> 如果不指定 model，ChatOpenAI 默认会使用 "gpt-3.5-turbo" 模型。



输出：

```
Sure, here's one for you:

Why do software engineers prefer dark mode?

Because light attracts bugs!
```



## LangChain 提示词模板

通常我们都是这样写提示词：



```python
language = "Python"
prompt = f"What is the best way to learn coding in {language}?"
print(prompt) # What is the best way to learn coding in Python?
```



### 为什么使用LangChain

但为什么不直接使用 f-string 来进行提示模板化呢？而改用 LangChain 的提示模板则能让你轻松做到以下几点：

- 验证你的提示输入
- 通过组合将多个提示整合在一起
- 定义自定义选择器，将 k-shot 示例注入到提示中
- 从 .yml 和 .json 文件中保存和加载提示
- 创建在生成时可以执行额外代码或指令的自定义提示模板



### LangChain 表达式语言（LCEL）

LangChain 表达式语言（LCEL）
 “|” 管道运算符是 LangChain 表达式语言（LCEL）的关键组件，它允许你在数据处理流水线中将不同的组件或可运行单元串联在一起。
 在 LCEL 中，“|” 运算符类似于 Unix 管道运算符：它将一个组件的输出作为输入传递给链中下一个组件，从而让你可以轻松地连接和组合不同的组件，创建出复杂的操作链。例如：

```python
chain = prompt | model
```

这里，“|” 运算符用于将 prompt 和 model 组件串联在一起。prompt 组件的输出会传递给 model 组件。这种链式机制使你可以从基本组件构建复杂的链，并实现数据在处理流水线各阶段之间的无缝流动。

另外，顺序非常重要，理论上你也可以构造如下链：

```python
bad_order_chain = model | prompt
```

但在调用 invoke 函数时会产生错误，因为 model 返回的值与 prompt 所期望的输入不兼容。

接下来，让我们使用提示模板创建一个商业名称生成器，该生成器将返回五到七个相关的商业名称。



```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)

template = """
You are a creative consultant brainstorming names for businesses.
You must follow the following principles:
{principles}
Please generate a numerical list of five catchy names for a start-up in the
{industry} industry that deals with {context}?
Here is an example of the format:
1. Name1
2. Name2
3. Name3
4. Name4
5. Name5
"""

model = ChatOpenAI(
    temperature=0.6,
    model="gpt-4o",  # Correctly specify the model name (ensure the model name is valid)
    api_key=userdata.get('OPENAI_API_KEY')
)

system_prompt = SystemMessagePromptTemplate.from_template(template)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt])
chain = chat_prompt | model

result = chain.invoke({
    "industry": "medical",
    "context": '''creating AI solutions by automatically summarizing patient records''',
    "principles": '''1. Each name should be short and easy to remember.
2. Each name should be easy to pronounce.
3. Each name should be unique and not already taken by another company.'''
})

print(result.content)

```

```
1. MedBrief
2. HealthSynth
3. RecordWise
4. SummarAIze
5. ChartGenius
```



## 在聊天模型中使用 PromptTemplate
LangChain 提供了一种更传统的模板，称为 PromptTemplate，它需要传入 input_variables 和 template 参数。

输入：

```python
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_openai.chat_models import ChatOpenAI

prompt = PromptTemplate(
    template='''You are a helpful assistant that translates {input_language} to {output_language}.''',
    input_variables=["input_language", "output_language"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
chat = ChatOpenAI()
chat.invoke(system_message_prompt.format_messages(
    input_language="English", output_language="French"))
```

输出：

```
AIMessage(content="Vous êtes un assistant utile qui traduit l'anglais en français.", additional_kwargs={}, example=False)
```
