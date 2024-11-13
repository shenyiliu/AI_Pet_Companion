## 项目介绍
- 这个项目是一个有趣的互动软件，结合了语音识别和人工智能技术。你可以通过语音与一个可爱的虚拟桌宠交流，用它来控制电脑、执行任务（如打开应用或搜索信息）。
- 该项目为[2024 AI+硬件创新大赛](https://competition.atomgit.com/competitionInfo?id=31577e662cba8dce522bb91b959e347e) 参赛项目，所选赛题为：“赛题一：基于成熟硬件（手机、PC、平板、汽车等），打造端云协同的AI应用”。

### 主要功能
1. 心理疗愈师：内置一位二次元风格的心理疗愈师角色，能够通过语音和文字与用户进行交流，提供情感支持和心理疏导。
2. 二次元动漫风格：采用Live2D技术，使虚拟角色更加生动活泼，增强用户的沉浸感和互动体验。
3. 控制电脑：用户可以通过语音命令来控制电脑，例如打开应用、播放音乐等，简化操作流程。
4. 调用工具：集成多种实用工具，如日程管理、提醒事项等，方便用户日常生活和工作。 

这款软件旨在为用户提供一个温馨、有趣且实用的互动平台，帮助他们在忙碌或压力大的时候得到放松和心理上的慰藉。无论是需要情绪支持还是简单的日常操作，都能在这个虚拟伙伴的帮助下变得更加轻松愉快。

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

## 部署工作
### 第一步：部署ASR(Automatic Speech Recognition 自动语音识别)服务
1. 使用anaconda创建并激活一个虚拟环境，比如可以叫`openvino`，可以参考下面的命令。
    ```shell
    conda create --name openvino python=3.10.15 -y
    conda activate openvino
    ```
2. 安装依赖
    ```bash
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    ```
3. 输入下面这条命令验证openvino，观察其支持的设备，理论上应该会输出`['CPU', 'GPU', 'NPU']`，如果打印结果缺少GPU或者NPU，请检查驱动是否正常。
    ```shell
    python -c "import openvino as ov; core = ov.Core(); print(core.available_devices)"
    ```
4. 模型转换，转原始模型为openvino需要的模型格式。
    ```bash
    optimum-cli export openvino --model download/Qwen2.5-7B-Instruct --weight-format int4 --task text-generation-with-past output/ov-qwen2.5-7b-instruct-int4
    ```
5. 启动ASR
    ```bash
    python ASR/Voice.py
    ```

### 第二步：部署TTS(Text To Speech 语音合成)服务
1. 采用[GPT-SoVITS-Inference](https://github.com/X-T-E-R/GPT-SoVITS-Inference)开源项目。
2. 从huggingface下载一键懒人包，[点击下载](https://huggingface.co/XTer123/GSVI_prezip/resolve/main/GSVI-2.2.4-240318.7z)。网络不畅的，可以使用hf-mirror.com，[点击下载](https://hf-mirror.com/XTer123/GSVI_prezip/resolve/main/GSVI-2.2.4-240318.7z)。
3. 解压下载的模型，将解压后的`GPT-SoVITS-Inference`文件夹放到本项目根目录。
4. 启动该项目。
    ```bash
    cd GPT-SoVITS-Inference
    runtime\python.exe ./Inference/src/tts_backend.py
    ```

### 第三步：部署卡通人
1. 采用[https://github.com/zenghongtu/PPet?tab=readme-ov-file](https://github.com/zenghongtu/PPet?tab=readme-ov-file)项目，点击一下人物即可开始录音，再点击一下停止录音。
2. 使用anaconda创建并激活一个虚拟环境，比如可以叫`live2d`，可以参考下面的命令。
    ```bash
    conda create -n Live2d python=3.10 -y
    conda activate Live2d
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

## 项目演示
- 截图
- gif图
