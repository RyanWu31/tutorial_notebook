# Tutorial_notebook 
20240107for Shanghai AI lab
# 目录
- [tutorial_notebook](#tutorial_notebook)
- [第一节](#第一节)
  - [笔记](#笔记)
- [第二节 轻松玩转书生·浦语大模型趣味 Demo](#第二节-轻松玩转书生浦语大模型趣味-demo)
  - [基础作业](#基础作业)
    - [1.300字小故事 InternLM-Chat-7B 模型生成 300 字的小故事](#1300字小故事-internlm-chat-7b-模型生成-300-字的小故事)
    - [2.下载20b 使用 huggingface_hub python 包，下载 InternLM-20B 的 config.json 文件到本地](#2下载20b-使用-huggingface_hub-python-包下载-internlm-20b-的-configjson-文件到本地)
  - [进阶作业](#进阶作业)
    - [1.lagent demo Lagent 工具调用 Demo 创作部署](#1lagent-demo-lagent-工具调用-demo-创作部署)
    - [2.浦语·灵笔的图文理解及创作部署](#2浦语灵笔的图文理解及创作部署)
  - [笔记](#笔记)
- [第 3 节 基于 InternLM 和 LangChain 搭建你的知识库](#第-3-节-基于-internlm-和-langchain-搭建你的知识库)
  - [基础作业](#基础作业-1)
  - [进阶作业](#进阶作业-1)
  - [笔记](#笔记-1)
- [第4节 XTuner 大模型单卡低成本微调实战](#第4节-xtuner-大模型单卡低成本微调实战)
  - [基础作业](#基础作业-2)
  - [笔记](#笔记-2)
- [第5节 LMDeploy 的量化和部署](#第5节-lmdeploy-的量化和部署)
  - [基础作业 使用 LMDeploy 以本地对话、网页Gradio、API服务中的一种方式部署 InternLM-Chat-7B 模型，生成 300 字的小故事（需截图）](#基础作业-使用-lmdeploy-以本地对话网页gradioapi服务中的一种方式部署-internlm-chat-7b-模型生成-300-字的小故事需截图)
- [第6节 OpenCompass 大模型评测](#第6节-opencompass-大模型评测)
  - [基础作业 使用 OpenCompass 评测 InternLM2-Chat-7B 模型在 C-Eval 数据集上的性能](#基础作业-使用-opencompass-评测-internlm2-chat-7b-模型在-c-eval-数据集上的性能)

# 第一节 

## 笔记
大模型通常是指机器学习和人工智能领域中的模型，其特点是具有庞大的参数数量和强大的计算能力。这些模型通过大规模数据训练，通常包含数十亿乃至数千亿个参数，常见的结构包括Transformer、BERT、GPT等。它们在自然语言处理、计算机视觉、语音识别等领域表现出卓越性能，能够理解复杂的数据特征和关系。

开源项目InternLM是一个轻量级的训练框架，旨在支持大模型训练，同时减少对依赖的需求。它在大型GPU集群上进行预训练，并在单个GPU上进行微调，具有出色的性能优化，可实现近90%的加速效率。基于InternLM，上海人工智能实验室发布了两个开源的预训练模型：InternLM-7B和InternLM-20B。

此外，Lagent是一个开源的大语言模型智能体框架，支持将大语言模型转化为多种智能体类型，并提供了相关工具以增强大语言模型的能力。

浦语·灵笔是基于书生·浦语大语言模型开发的视觉-语言大模型，具备强大的图文理解和创作能力，可实现图像到文本和文本到图像的双向转换。它可以用于创作图文推文，同时识别图像中的物体并生成相应的文本描述。


# 第二节 轻松玩转书生·浦语大模型趣味 Demo

## 基础作业
### 1.300字小故事 InternLM-Chat-7B 模型生成 300 字的小故事
本机
![5RMP`S$W9}Y_ZHU}@Q5B}_M](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/598100a4-89d5-4e23-a7b9-9124bf02dd8e)
webdemo

![}XO2Q3T%Y(D`UT99L~X)_KE](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/ae46a2d9-e9fb-47d6-96e3-b4e6cf22248c)
### 2.下载20b  使用 huggingface_hub python 包，下载 InternLM-20B 的 config.json 文件到本地
![4$JH3Z(W8MRBL8BS`6X6C`H](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/6ca11a38-98b6-4c21-849d-11e1a91b5689)
![2PCA}7PZU0QJK(_)VM4(B~3](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/2db1dc3b-1013-4d4e-96cc-db2e1d6fa023)
![M4RH4XW@_I8VRYG`FAMVUBI](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/c3c558ec-f4cf-4b6d-ab20-46a76207d20f)



<pre>
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download internlm/internlm-20b --local-dir ./')
```
</pre>


## 进阶作业


### 1.lagent demo  Lagent 工具调用 Demo 创作部署
![X@28Y_ (JGZ(~@@UZ9B7XQP](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/b446cd8a-c341-40de-bf0d-d2a5dc9f7728)
![)$V1I4)OE91LNFNVWFCJS$S](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/4087db96-7bde-4bb6-8088-d1692f08ad68)

### 2.浦语·灵笔的图文理解及创作部署

![A58S7TC6 9O5(B34PM~`DNE](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/0dff248e-71a9-416e-9bb9-c56fcb92dd51)
![AMA%_ 5F QPN%XT~3(O%UGI](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/438eca16-0e85-4a92-a96f-6b57d292f558)

## 笔记
如果git clone失败，可以用gitee
<pre>
git clone https://github.com/open-compass/opencompass
fatal: unable to access 'https://github.com/open-compass/opencompass/': Received HTTP code 503 from proxy after CONNECT   
git clone https://gitee.com/open-compass/opencompass
</pre>
如果huggingface下载失败，可以用huggingface镜像

InternLM2系列模型是一系列具有卓越性能的开源模型，其主要特点如下：

200K上下文窗口：InternLM2模型具有长达200K的上下文窗口，在长文本任务如LongBench和L-Eval中表现出色，能够高效地处理大规模文本数据。

综合性能卓越：相较于上一代模型，在各个方面性能显著提升，尤其在推理、数学、编程、聊天交互、指令跟随以及创造性写作等方面，表现领先于类似规模的开源模型。在某些评估中，InternLM2-Chat-20B甚至可能与ChatGPT（GPT-3.5）媲美甚至超越。

代码解释和数据分析：InternLM2-Chat-20B通过代码解释支持GSM8K和MATH等任务，与GPT-4在性能上相当。此外，InternLM2-Chat还提供了数据分析功能。

强大的工具使用：InternLM2在指令跟随、工具选择和反思等方面具有更好的工具利用能力，支持更多类型的智能体和多步骤工具调用，适用于复杂任务。

模型大小：InternLM2系列包含两种模型规模，即7B和20B。7B模型适用于研究和应用，而20B模型更强大，能够支持更复杂的场景。

此外，该模型系列的性能表现也得到了详细的评估，包括客观评估、长文本评估、数据污染评估、智能体评估和主观评估等多个方面。

总之，InternLM2系列模型是一组性能卓越的开源模型，具有广泛的应用潜力，可用于各种任务，但仍然需要注意模型的潜在限制，如可能产生偏见、歧视或其他有害内容的风险。


7b不如chatglm3， 20b比百川13b强





# 第 3 节	基于 InternLM 和 LangChain 搭建你的知识库
## 基础作业
本机复现
![5SARWN_P_UTH3{N7SJ)5`RX](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/e1e5269f-9c81-46e8-b5b0-abf4d0c9ca06)
## 进阶作业
openxlab部署
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/a5960bc6-9554-4afd-99be-693c898447b9)
https://openxlab.org.cn/apps/detail/RyanWu31/fin_Chat
https://github.com/RyanWu31/fin_Chat
先放这回头改
不知道这个sentence transofrmer和大模型权重应该放哪
![W86Z}HE5`NG{8(D3G2H`}NY](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/d34908a9-f217-4b28-937b-bbb2bffef5b4)
![J7M}93URS2L0@X}AV{$LUM6](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/49e20ff9-c5f6-4005-bd2a-2b8e8d3b7454)
![9R~J VU(1UUY1CC M~YJUCK](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/b36a5e19-b1f7-4809-ae27-2131c0fa1204)

https://openxlab.org.cn/models/detail/RyanWu31/Financial_Dialogue



## 笔记
如果在一个局域网内，相同端口可以在不同机器上直接运行，7680端口我没做映射直接就能跑，在本机做的web_demo
这个地方直接换模型不能运行，不知道为啥，用chatglm3推理不出来
这个问题解决了，就是response的格式不一样，例如llama有统一的resposon格式


LangChain是一个用于自然语言处理任务的平台，它允许用户通过简化流程来创建和部署NLP模型，同时提供了一系列有用的工具和资源。主要功能包括：

模型训练：LangChain提供了用于训练NLP模型的工具和框架，使用户能够根据自己的需求进行模型训练。您可以选择不同的预训练模型、数据集和任务类型来定制您的模型。

模型评估：LangChain支持模型性能的客观评估，帮助用户了解模型在不同任务上的表现。您可以使用内置的评估指标来量化模型的性能。

模型部署：LangChain还允许用户将他们的训练好的模型部署到生产环境中，以便进行实际应用。这使用户可以将其NLP模型应用于各种应用程序，如聊天机器人、文本分类、情感分析等。



![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/a52f2180-9703-40a3-9198-62e15025a2cb)
### 知识库就是一些文本，作为模型的先验知识
怎么找知识库？
需要遵守一些格式吗？直接把文字爬下来放那就行？



# 第4节 XTuner 大模型单卡低成本微调实战
## 基础作业
这里效果不太好，把代码放上来以后调
<pre>
```python
(xtuner0.1.9) dell@dell-PowerEdge-T640:~/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant$ NPROC_PER_NODE=2 xtuner train /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2

(xtuner0.1.9) dell@dell-PowerEdge-T640:~/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant$ xtuner convert pth_to_hf /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/internlm_chat_7b_qlora_oasst1_e3_copy.py /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_20.pth /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/work_dirs/hf

(xtuner0.1.9) dell@dell-PowerEdge-T640:~/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant$ xtuner convert pth_to_hf /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/internlm_chat_7b_qlora_oasst1_e3_copy.py /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_20.pth /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/work_dirs/hf

(xtuner0.1.9) dell@dell-PowerEdge-T640:~/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant$ xtuner convert merge /home/dell/.cache/modelscope/hub/Shanghai_AI_Laboratory/internlm-chat-7b /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/work_dirs/hf /home/dell/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/config/work_dirs/hf_merge --max-shard-size 2GB



ll/remote/Big_model/Shanghai_AI_lab_2024tutorial/4/xtuner019/xtuner/personal_assistant/InternLM/web_demo.py --server.address 127.0.0.1 --server.port 6006

  
</pre>
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/432ccc8a-4eb0-4d25-bb22-70474374df9f)
训练个20epoch还不行？
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/8014bc81-a575-4306-9079-f0dc679728d8)
啥情况啊，根本不带变化的
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/6cf8b0bb-2718-494c-9925-012cbaa745c9)
不如直接prompt来的快
## 笔记
这节问题比较多，先是小助手fine-tune没效果，说是主人的名字不能用英文，然后改成中文还是没效果。看了一下群里说的，要改max_length，还是没效果。又将epoch改成5，还是没效果，不知道为啥
可能是因为用的多卡加速？就用的两块A100,5个epoch 1个小时就训练完了。
NPROC_PER_NODE=2 xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py


# 第5节 LMDeploy 的量化和部署

## 基础作业 使用 LMDeploy 以本地对话、网页Gradio、API服务中的一种方式部署 InternLM-Chat-7B 模型，生成 300 字的小故事（需截图）
本地对话
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/1a5e5db1-30a6-4788-a539-d8c19e6ab0ec)
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/63b2def1-259f-4129-8767-23ccc2889620)


# 第6节 OpenCompass 大模型评测

## 基础作业 使用 OpenCompass 评测 InternLM2-Chat-7B 模型在 C-Eval 数据集上的性能

![}{}_CM_ 2SWVEJ AWP9KGAM](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/270e07db-aaed-4f19-853f-b5ca04ff9721)

![8SI%0G70N2LEU_AEYJ_UPH7](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/18121f99-68f0-4b07-afc7-c8904312cc2f)

![076A%NVZ5AEX4E O}BDKW`X](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/2de99a25-7505-48a1-915a-1cec31f6f905)

## 笔记


OpenCompass是一个多功能的NLP模型评估工具，具有以下主要特点：

客观性能评估：OpenCompass提供了多种客观性能指标，用于评估NLP模型的性能。这些指标可以用于不同类型的任务，如文本生成、文本分类、机器翻译等。

多任务支持：它支持多种不同的NLP任务，包括但不限于多选题、文本生成、回归任务等。这使得用户可以在各种任务上进行全面的性能评估。

自定义评估指标：OpenCompass还允许用户定义自己的评估指标，以便根据具体需求进行评估。这对于特定任务和应用程序非常有用。

结果可视化：OpenCompass提供了可视化工具，以直观地呈现评估结果，帮助用户更容易地理解模型的性能。

使用OpenCompass时应注意的事项：

数据质量：评估结果的质量取决于使用的数据集的质量。确保使用高质量的数据集进行评估，以获得准确的性能指标。

任务选择：根据您的需求选择合适的任务进行评估。不同的任务可能需要不同的评估指标。

模型选择：在评估时，选择合适的NLP模型是至关重要的。不同模型在不同任务上可能表现出不同的性能。

细致分析：不仅要查看总体性能指标，还应该进行细致的分析，以了解模型在不同方面的表现。这有助于发现模型的潜在问题。

比较和基准：将模型的性能与其他模型或基准进行比较，以便更好地了解模型在领域中的位置。

持续监控：性能评估应该是一个持续的过程，随着时间的推移，模型的性能可能会发生变化。因此，建议定期重新评估模型。


