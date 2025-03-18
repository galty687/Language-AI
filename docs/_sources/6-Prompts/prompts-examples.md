# 提示词案例分析



## 使用LLM做分类

**分类**指的是预测给定数据点或样本所属的类别或类别。这是机器学习中的常见任务，其中模型被训练以根据学习到的模式为未标注数据分配预定义的标签。

大语言模型（LLMs）在分类任务中非常强大，即使在提示中仅提供零个或少量示例，也能表现出色。为什么呢？这是因为像 GPT-4 这样的 LLM 之前已经在庞大的数据集上进行了训练，并且具备一定的推理能力。

在使用 LLM 解决分类问题时，主要有两种策略：零样本学习（Zero-shot learning）和少样本学习（Few-shot learning）。

### 情感分类任务提示词


```md
Given the statement, classify it as either "Compliment", "Complaint", or "Neutral":

Examples:

   1. "The sun is shining." → Neutral
   2. "Your support team is fantastic!" → Compliment
   3."I had a terrible experience with your software." → Complaint

Classification Rules:

   1. Only return the single classification word.
   2. The response should be either "Compliment", "Complaint", or "Neutral".
   3. Perform the classification on the text enclosed within """ delimiters.

Statement to Classify:

"""The user interface is intuitive."""
Classification: 
```

### 在OpenAI中应用

```python
from openai import OpenAI
import os

from google.colab import userdata


# 初始化 OpenAI 客户端
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# 基础提示模板
base_template = """
Given the statement, classify it as either "Compliment", "Complaint", or "Neutral":
1. "The sun is shining." - Neutral
2. "Your support team is fantastic!" - Compliment
3. "I had a terrible experience with your software." - Complaint

You must follow the following principles:
- Only return the single classification word. The response should be either
  "Compliment", "Complaint", or "Neutral".
- Perform the classification on the text enclosed within ''' delimiters.

'''{content}'''
Classification:
"""

# 存储响应结果
responses = []

# 生成多个分类响应
for i in range(3):
    response = client.chat.completions.create(
        model="gpt-4o",
      	temperature=0,
        messages=[
            {
                "role": "system",
                "content": base_template.format(
                    content="""Outside is rainy, but I am having a great day. 
                    I just don't understand how people live, I'm so sad!"""
                ),
            }
        ],
    )
    responses.append(response.choices[0].message.content.strip())

# 找到最常见的分类
def most_frequent_classification(responses):
     # 使用字典统计每种分类的出现次数
    count_dict = {}
    for classification in responses:
        count_dict[classification] = count_dict.get(classification, 0) + 1

    # 返回出现次数最多的分类
    return max(count_dict, key=count_dict.get)
  
  
most_frequent_classification(responses)
```

输出：

```
Neutral
```

> 在实验过程中，默认温度的情况下，偶然也会遇到输出 Complaint 的情形，如果需要复现可以设置温度为0： `temperature=0,`

### 其他应用

- **客户评价分类**  
  - 将用户评论分类为“正面”、“负面”或“中性”。  
  - 进一步深入分析，可以识别子主题，如“可用性”、“客户支持”或“价格”。

- **电子邮件过滤**  
  - 检测电子邮件的意图或目的，并将其分类为“咨询”、“投诉”、“反馈”或“垃圾邮件”。  
  - 这有助于企业合理安排回复优先级，并高效管理沟通。

- **社交媒体情感分析**  
  - 监测社交媒体平台上的品牌提及和情感倾向。  
  - 将帖子或评论分类为“赞扬”、“批评”、“询问”或“中性”。  
  - 通过分析公众舆论，调整营销或公关策略。

- **新闻文章分类**  
  - 面对每天大量生成的新闻内容，LLMs 可以根据主题或话题对文章进行分类，例如“政治”、“科技”、“环境”或“娱乐”。

- **简历筛选**  
  - 对于 HR 部门面临的大量简历，可根据预定义标准进行分类，例如“合格”、“资历过高”或“资历不足”。  
  - 也可以按专业领域分类，如“软件开发”、“市场营销”或“销售”。



## 打分员

人工的评估方法中，通常采用简单的**点赞/点踩**的评分机制，以确定模型的回答是否符合我们的预期。然而，人工评分既昂贵又繁琐，需要合格的评估者来判断回答的质量或识别错误。一个越来越常见的方法是使用更高级的大语言模型（LLM）来评估较小模型的回答质量。

```python
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Store responses
responses = []

# Generate responses
for i in range(10):
    # Concise if even, verbose if odd
    style = "concise" if i % 2 == 0 else "verbose"
    
    if style == "concise":
        prompt = f"Return a {style} answer to the following question: What is the meaning of life?"
    else:
        prompt = "Return an answer to the following question: What is the meaning of life?"
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using GPT-3.5 Turbo for this example
        messages=[{"role": "user", "content": prompt}]
    )
    
    responses.append(response.choices[0].message.content.strip())

# System prompt for rating conciseness
system_prompt = """You are assessing the conciseness of a response from a chatbot.
You only respond with a 1 if the response is concise, and a 0 if it is not.
"""

# Store ratings
ratings = []

# Evaluate responses
for idx, response in enumerate(responses):
    rating = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": response}
        ]
    )
    
    ratings.append(rating.choices[0].message.content.strip())

# Print results
for idx, rating in enumerate(ratings):
    style = "concise" if idx % 2 == 0 else "verbose"
    print(f"Style: {style}, Rating: {rating}")

```

输出：

```
Style: concise, Rating: 1
Style: verbose, Rating: 0
Style: concise, Rating: 1
Style: verbose, Rating: 0
Style: concise, Rating: 1
Style: verbose, Rating: 0
Style: concise, Rating: 1
Style: verbose, Rating: 0
Style: concise, Rating: 1
Style: verbose, Rating: 0

```

这个脚本是一个 Python 程序，它通过与 OpenAI API 的交互来生成并评估基于简洁性的回答。下面是具体解释：

1. **responses = []** 创建了一个名为 responses 的空列表，用来存储 OpenAI API 生成的回答。
2. **for 循环运行 10 次**，每次迭代都会生成一个回答。
3. 在循环内部，根据当前的迭代次数（i）确定回答的风格。偶数迭代为“简洁”，奇数迭代为“详细”。
4. 根据所选风格，格式化提示字符串，用不同方式（简洁或详细）询问“生命的意义是什么？”。
5. **response = client.chat.completions.create(...)** 向 OpenAI API 发送请求，根据提示生成回答，此处使用的模型为 “gpt-3.5-turbo”。
6. 将生成的回答去除首尾空白后添加到 responses 列表中。
7. **system_prompt = """You are assessing..."""** 设置了一个用于评估生成回答简洁性的系统提示。
8. **ratings = []** 初始化一个空列表，用来存储简洁性评分。
9. 另一个 for 循环遍历 responses 列表中的每个回答。
10. 对于每个回答，脚本将回答和 system_prompt 一并发送给 OpenAI API，请求进行简洁性评估。这次使用的模型为 “gpt-4”。
11. 将评估得到的评分（1 表示简洁，0 表示不简洁）去除空白后添加到 ratings 列表中。
12. 最后一个 for 循环遍历 ratings 列表，对每个评分打印出回答的风格（“简洁”或“详细”）及其对应的简洁性评分。

## Meta Prompt

元提示是一种技术，涉及创建文本提示，而这些文本提示又能生成其他文本提示。然后，这些文本提示被用来生成在图像、视频以及更多文本等多种媒介中的新资源。

为了更好地理解元提示，我们以借助 GPT-4 撰写儿童书籍为例。首先，你指示大型语言模型生成儿童书籍的文本内容。之后，你通过指令 GPT-4 生成适用于图像生成模型的提示，从而启动元提示。这可能意味着根据书中的情节生成情景描述或具体场景的提示，随后这些提示可以传递给 Midjourney 或 Stable Diffusion 等 AI 模型。

因此，这些图像生成模型能够与由 AI 创作的儿童故事保持和谐一致地提供相应的图像。



输入 ChatGPT o3-mini：

```md
Act as a GPT Prompt Engineer, you are solely responsible for producing highly effective large language model text prompts given a task.

You must follow the following principles:
- Return only a single prompt, never return the output of a prompt.
- You must follow the task properly.
- You must either refine the prompt or create a new prompt depending upon the Task.
- Bullet point out any important features so that another Prompt Engineer can easily copy inputs into the prompt.

Task: You must create a copywriting guide ChatGPT several blog posts. Include relevant features within the ChatGPT prompt that will allow ChatGPT to imitate the writer. You must decide what the most important features are to mimic the writer's style etc. You will only be applying this style to new blog posts.

Blog post text: You need a variety of soft skills in addition to technical skills to succeed in the technology sector. Soft skills are used by software professionals to collaborate with their peers...
```

输出：

```
You are an expert copywriting guide tasked with generating several new blog posts. Your objective is to create engaging, informative, and authentic blog posts that capture and reflect the unique style of a seasoned software professional writer. The writer emphasizes the importance of soft skills alongside technical expertise. Use the following guidelines to shape your output:

    Voice & Tone: Maintain a professional yet approachable tone that conveys expertise in both technology and interpersonal skills.
    Style Mimicry: Imitate the writer’s style as seen in the sample text ('You need a variety of soft skills in addition to technical skills to succeed in the technology sector. Soft skills are used by software professionals to collaborate with their peers...'). Focus on clarity, concise language, and a balanced blend of technical insights with human-centered advice.
    Content Focus: Each blog post should highlight how soft skills complement technical abilities, offering practical advice, examples, and actionable insights that appeal to professionals in the technology sector.
    Structure & Flow: Ensure the blog posts are logically structured, include engaging headlines, and smoothly transition between technical discussions and soft skills insights.
    Consistency: Apply this style consistently across multiple posts, ensuring that each post feels like a natural extension of the writer’s voice and perspective.

Using the above instructions, create several new blog posts that reflect the established style and focus on the integration of soft skills and technical skills in the tech industry.
```

用提示词后的输出：

```md
Below are several blog posts crafted to capture the unique style of a seasoned software professional writer who champions the integration of soft skills with technical expertise.

---

### Blog Post 1: The Interplay of Soft Skills and Technical Expertise

#### Introduction
In today’s technology sector, success isn’t determined solely by code quality or technical know-how. You need a variety of soft skills in addition to technical skills to succeed in the technology sector. Soft skills are used by software professionals to collaborate with their peers, navigate project challenges, and drive innovation. In this post, we explore how balancing technical acumen with interpersonal abilities can propel your career forward.

#### The Technical Foundation
Modern software development demands proficiency in programming languages, frameworks, and system design. However, the technical foundation is just the beginning. Projects often require integrating diverse technologies, troubleshooting complex issues, and continuously learning new paradigms. Mastering these technical skills lays the groundwork for professional growth.

#### The Soft Skills Edge
What sets great professionals apart is the ability to communicate ideas clearly, negotiate conflicts, and inspire teams. Whether you’re explaining a new algorithm to a non-technical stakeholder or collaborating on a high-pressure project, soft skills make the difference. For instance, active listening and empathy can transform a challenging team dynamic into a productive brainstorming session.

#### Practical Strategies for Integration
- **Collaborative Problem Solving:** Schedule regular team meetings to discuss technical challenges and encourage input from all members. This approach not only solves problems faster but also builds trust.
- **Continuous Learning:** Attend workshops or webinars focused on both technical updates and communication techniques. Blending these perspectives ensures you stay ahead in both areas.
- **Feedback Culture:** Establish a routine of peer feedback where technical critique is balanced with constructive personal insights. This dual approach fosters both innovation and professional growth.

#### Conclusion
Balancing technical expertise with strong soft skills is not an option but a necessity in the evolving tech landscape. By nurturing both aspects, you can unlock a well-rounded professional profile that stands out in any team. Embrace the synergy between hard skills and soft skills to drive your career forward.

---

### Blog Post 2: Communication in Code: Why Soft Skills Are as Crucial as Technical Know-How

#### Introduction
In the fast-paced world of technology, technical skills often get all the spotlight. Yet, it’s the soft skills that enable software professionals to bridge the gap between brilliant ideas and practical solutions. In this post, we delve into why effective communication is just as vital as writing clean code.

#### The Communication Challenge in Tech
Developers and engineers are experts in their domains, but conveying complex technical ideas can be challenging. Miscommunication not only leads to errors but can also create friction within teams. This is where a focus on soft skills, such as clarity in communication and active listening, proves invaluable.

#### Examples from the Field
Consider a scenario where a developer must explain a new feature to a project manager with limited technical background. Using clear, relatable language, complemented by visual aids or analogies, can demystify complex concepts. This not only facilitates better decision-making but also strengthens team cohesion.

#### Actionable Tips for Enhancing Communication
- **Practice Clarity:** Break down technical jargon into simple terms. Use diagrams and real-life examples to support your points.
- **Engage in Active Listening:** Encourage team discussions where every voice is heard. This helps in capturing diverse perspectives and finding innovative solutions.
- **Feedback Loops:** Regularly solicit feedback on your communication style. Adjusting your approach based on input can lead to continuous improvement in how you share ideas.

#### Conclusion
Strong communication skills are essential for translating technical expertise into effective collaboration. By honing your ability to articulate ideas clearly and listen actively, you’ll not only enhance your technical contributions but also foster a more productive, innovative work environment.

---

### Blog Post 3: Leading with Empathy in Tech: Merging Technical Rigor with Human-Centered Leadership

#### Introduction
The technology industry is renowned for its focus on metrics, algorithms, and scalable solutions. However, true leadership in tech goes beyond technical achievements. Leading with empathy and integrating soft skills into your daily interactions can create a more inclusive and dynamic workplace.

#### The Role of Empathy in Technical Teams
Empathy might seem like an unconventional tool in the world of software development, but it’s critical for understanding team dynamics. When leaders practice empathy, they create a culture where every team member feels valued and understood. This fosters an environment where technical challenges are met with collective creativity and resilience.

#### Bridging the Gap Between Code and People
Imagine a scenario where a team faces a tight deadline, and tensions are running high. A leader who takes the time to understand each team member’s perspective can diffuse stress and refocus the group on common goals. By aligning technical objectives with personal well-being, projects not only succeed but also build long-term team loyalty.

#### Strategies for Empathetic Leadership
- **Regular Check-Ins:** Schedule one-on-one meetings with team members to understand their challenges and aspirations. This practice encourages open dialogue and personal growth.
- **Transparent Communication:** Share both successes and challenges with your team. Transparency builds trust and encourages collective problem-solving.
- **Invest in Professional Development:** Support opportunities for both technical training and soft skills development. Empower your team to grow in all aspects of their careers.

#### Conclusion
Incorporating empathy into technical leadership is a game changer. When you combine the analytical rigor of software development with the human touch of effective leadership, you create a team that is not only technically proficient but also deeply committed to collective success. Embrace empathy to lead with confidence and foster a culture of innovation and inclusivity.

---

Each of these blog posts is designed to resonate with technology professionals by offering clear, actionable advice while underscoring the critical role of soft skills in a tech-centric environment. Enjoy sharing these insights with your audience!
```

