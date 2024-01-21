# tutorial_notebook
20240107for Shanghai AI lab


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
# 第 3 节	基于 InternLM 和 LangChain 搭建你的知识库
## 基础作业
本机复现
![5SARWN_P_UTH3{N7SJ)5`RX](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/e1e5269f-9c81-46e8-b5b0-abf4d0c9ca06)
## 进阶作业
openxlab部署
![W86Z}HE5`NG{8(D3G2H`}NY](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/d34908a9-f217-4b28-937b-bbb2bffef5b4)
![J7M}93URS2L0@X}AV{$LUM6](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/49e20ff9-c5f6-4005-bd2a-2b8e8d3b7454)
![9R~J VU(1UUY1CC M~YJUCK](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/b36a5e19-b1f7-4809-ae27-2131c0fa1204)

https://openxlab.org.cn/models/detail/RyanWu31/Financial_Dialogue



## 笔记
如果在一个局域网内，相同端口可以在不同机器上直接运行，7680端口我没做映射直接就能跑，在本机做的web_demo
这个地方直接换模型不能运行，不知道为啥，用chatglm3推理不出来
这个问题解决了，就是response的格式不一样，例如llama有统一的resposon格式
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/a52f2180-9703-40a3-9198-62e15025a2cb)
## 知识库就是一些文本，作为模型的先验知识
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






