# tutorial_notebook
20240107for Shanghai AI lab


# 第二节 轻松玩转书生·浦语大模型趣味 Demo

## 基础作业
### 300字小故事
本机
![5RMP`S$W9}Y_ZHU}@Q5B}_M](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/598100a4-89d5-4e23-a7b9-9124bf02dd8e)
webdemo

![}XO2Q3T%Y(D`UT99L~X)_KE](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/ae46a2d9-e9fb-47d6-96e3-b4e6cf22248c)
### 下载20b
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



# 第 3 节	基于 InternLM 和 LangChain 搭建你的知识库
## 基础作业
本机复现
![5SARWN_P_UTH3{N7SJ)5`RX](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/e1e5269f-9c81-46e8-b5b0-abf4d0c9ca06)


## coding
如果在一个局域网内，相同端口可以在不同机器上直接运行，7680端口我没做映射直接就能跑，在本机做的web_demo
这个地方直接换模型不能运行，不知道为啥，用chatglm3推理不出来
![image](https://github.com/RyanWu31/tutorial_notebook/assets/110294962/a52f2180-9703-40a3-9198-62e15025a2cb)
## 知识库就是一些文本，作为模型的先验知识
怎么找知识库？
需要遵守一些格式吗？直接把文字爬下来放那就行？
