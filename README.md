## 项目介绍
- 端侧AI桌宠助手，基于Intel AIPC的本地算力，结合Live2D二次元角色界面，提供个性化记忆对话和隐私保护的端侧AI桌宠助手。
- 该项目为[2024 AI+硬件创新大赛](https://competition.atomgit.com/competitionInfo?id=31577e662cba8dce522bb91b959e347e) 参赛项目，所选赛题为：“赛题一：基于成熟硬件（手机、PC、平板、汽车等），打造端云协同的AI应用”。

### 亮点介绍：
1. 我们采用心理咨询领域的专业知识，并结合胡桃角色的对话风格，对qwen2.5 7B模型进行了细致的微调，以提升其对话能力。

2. 通过ipex-ollama技术，我们将微调后的模型部署在Intel集成图形处理器（iGPU）上，以充分利用本地硬件资源，实现高效的模型运行。

3. 利用Langchain的长短期记忆模块，并结合本地向量数据库存储对话信息，我们让AI桌宠更加理解用户，提供更加个性化的交互体验。

4. 通过整合BERT意图分类与实体识别模型，并借助OpenVino技术及AIPC上的NPU硬件，我们成功打造了一个低延迟（10ms左右）、高效能的智能控制系统。该系统能够无缝操控多种设备，包括调节电脑亮度、音量、摄像头等，涵盖超过十项实用功能。这一创新为用户带来了近似端侧贾维斯的智能化体验，极大地提升了操作便捷性与响应速度。

5. 结合自动语音识别（ASR）、大型语言模型（LLM）和文本转语音（TTS）技术，我们通过特定关键词“胡桃胡桃”实现语音唤醒功能，自动识别用户对话内容，打造出具有胡桃二次元角色特色的语音对话体验，并实现语音控制工具调用的能力。

6. 参考OpenVINO对OpenVoice-v1的适配指南，我们顺利完成了OpenVoice-v2的适配工作。在此基础上，我们进一步运用NNCF对OpenVINO v2进行了4/8 bit模型量化处理，不仅显著提升了语音生成的质量，还大幅降低了响应延迟，从而实现了更高效、更迅捷的语音生成体验。依托OpenVoice-V2强大的语音克隆能力，我们能够基于任意一个参考音频和需要输出的文本，生成与参考音频高度匹配的语音，极大地提升了语音生成的可玩性和灵活性。

7. 端侧模型的实现确保了隐私数据的本地化处理，最大程度上保护了用户的隐私安全。

### 产品简介：
一款端侧AI桌宠助手，依托于Intel AIPC的强大本地计算能力，采用Live2D技术打造的二次元角色界面，为您的AI助手带来生动的视觉体验。这款助手能够持续记忆对话内容，实现深度个性化的AI交互体验。通过定制化的语音交互，它能够控制PC的多项功能，如调整屏幕亮度、音量以及摄像头设置。所有关键信息均存储于本地，确保您的隐私安全。此外，我们还可以针对不同角色IP定制语言风格和语音音色，以满足您的个性化需求。我们致力于创造一款完全在端侧运行的AI桌宠助手，让您的数字生活更加丰富多彩。
![logo](./images/logo.png)

### 主要功能
1. 微调的对话模型：基于心理咨询专业知识和胡桃角色的对话风格，微调了qwen2.5 7B模型，提升其对话能力。
2. 本地硬件部署：利用ipex-ollama技术，将微调后的模型部署在Intel集成图形处理器（iGPU）上，确保高效运行。
3. 个性化对话体验：通过Langchain的长短期记忆模块和本地向量数据库存储对话信息，提升AI桌宠的用户理解能力和互动体验。
4. 多工具控制：结合BERT意图分类和实体识别模型，实现低延迟控制多个工具（如调整电脑亮度、音量、摄像头等）功能，提供类似端侧贾维斯的智能体验。
5. 语音唤醒与控制：通过自动语音识别（ASR）、大型语言模型（LLM）和文本转语音（TTS）技术，支持语音唤醒（“胡桃胡桃”）及语音控制工具调用。
6. 优化语音生成速度：完成openvino对OpenVoice-v2的适配，提升语音生成效果以及大幅度降低了响应延迟。
7. 个性化语音克隆：只需提供一个参考音频（当前默认是胡桃的音频）和所需输出的文本，OpenVoice-V2就能生成与参考音频高度匹配的语音，无论是模仿名人声音、再现经典台词，还是创造全新的角色声音，都能轻松实现。
8. 隐私保护：端侧模型确保隐私数据本地处理，保护用户隐私安全。


## 准备工作
1. 需要英特尔AIPC，因为本项目需要借助openvino框架将模型离线运行在英特尔CPU/GPU/NPU平台。
2. 需要已安装GPU/NPU驱动。Windows用户可以通过ctrl+alt+del组合键，选择任务管理器，选择`性能`来查看是否有英特尔GPU和英特尔NPU。
3. 需要已安装anaconda。
4. 克隆本项目，进入项目路径。
    ```bash
    git clone https://github.com/shenyiliu/AI_Pet_Companion.git
    cd AI_Pet_Companion
    ```
5. 去huggingface下载[Qwen2.5-7B-Instruct](https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct)和[Qwen2-VL-7B-Instruct](https://hf-mirror.com/Qwen/Qwen2-VL-7B-Instruct)模型，放置到本项目的`download`目录。网络不畅的，可以使用hf-mirror.com。
6. 去魔搭下载[SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall)和[speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)，放置到本项目的`download`目录。
7. 使用anaconda创建并激活一个虚拟环境，比如可以叫`openvino`，可以参考下面的命令。
    ```shell
    conda create --name openvino python=3.10.15 -y
    conda activate openvino
    ```
8. 安装依赖
    ```bash
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    ```
9. 输入下面这条命令验证openvino，观察其支持的设备，理论上应该会输出`['CPU', 'GPU', 'NPU']`，如果打印结果缺少GPU或者NPU，请检查驱动是否正常。
    ```shell
    python -c "import openvino as ov; core = ov.Core(); print(core.available_devices)"
    ```

## 部署工作
### 第一步：部署LLM
- 数据生成以及模型lora微调过程 + 模型权重合并过程（待上传）
- 讲合并后的权重放置到xxxx

### 第二步：部署bert意图分类器
- 数据生成过程（待上传）
- [部署文档](./bert_tools/README.md)


### 第三步：部署ASR(Automatic Speech Recognition 自动语音识别)服务
4. 模型转换，转原始模型为openvino需要的模型格式。
    ```bash
    optimum-cli export openvino --model download/Qwen2.5-7B-Instruct --weight-format int4 --task text-generation-with-past output/ov-qwen2.5-7b-instruct-int4
    ```
5. 启动ASR
    ```bash
    python ASR/Voice.py
    ```


### 第四步：部署TTS(Text To Speech 语音合成)服务
- 采用[OpenVoice](https://github.com/myshell-ai/OpenVoice)开源项目的V2版本，使用Intel OpenVino进行推理加速。
- [部署文档](./open_voice_v2/README.md)

### 第五步：部署卡通人
1. 采用[https://github.com/zenghongtu/PPet?tab=readme-ov-file](https://github.com/zenghongtu/PPet?tab=readme-ov-file)项目
2. 可以复用第一步创建的openvino环境。
    ```bash
    conda activate openvino
    ```
3. 使用conda安装nodejs。
    ```bash
    conda install -c anaconda nodejs
    ```
4. 安装pnpm，并设置npm仓库为国内源，加快下载速度。
    ```bash
    npm install -g pnpm -i --registry=https://registry.npmmirror.com
    pnpm config set registry https://registry.npmmirror.com
    ```
5. 进入本项目的PPet文件夹，安装依赖。
    ```bash
    cd PPet
    pnpm i
    ```
6. 启动PPet项目。
   ```bash
   pnpm start
   ```

7. 可以通过语音直接进行对话，唤醒关键词为“胡桃胡桃”（注：需要电脑支持麦克风）

## 项目演示视频
- 待上传