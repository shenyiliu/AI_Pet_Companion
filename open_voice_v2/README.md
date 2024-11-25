### 简单说明
- 参考了[官方教程](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvoice/openvoice.ipynb)
1. 创建一个虚拟环境openvoice，顺便安装ffmpeg
   ```bash
   conda create -n openvoice python==3.10 -y
   conda activate openvoice
   conda install ffmpeg
   ```
2. 安装openvoice关联依赖
    ```bash
    pip install -r open_voice/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    ```

3. 安装MeloTTS和unidic
   ```bash
   pip install git+https://github.com/myshell-ai/MeloTTS.git
   # 下面这步可能需要`较好的网络`, 网络不好的可以参考该帖子：https://blog.csdn.net/Ppandaer/article/details/140045774
   python -m unidic download
   ```
4. 下载checkpoints_v2权重，解压后放`download/OpenVoice/`目录，[下载地址](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip)并解压
   - 下载中文tts放`download/OpenVoice/checkpoint_v2`目录，[huggingface下载地址](https://hf-mirror.com/myshell-ai/MeloTTS-Chinese)
   - 下载英文tts放`download/OpenVoice/checkpoint_v2`目录，[huggingface下载地址](https://huggingface.co/myshell-ai/MeloTTS-English-v2)
   - 网络不好的可以直接用我下载好的百度网盘。链接: https://pan.baidu.com/s/1sF-yBvYbgktLFdrlSdeDzQ?pwd=d744 提取码: d744
   - 存放位置参考
   ```bash
      checkpoints_v2   
      ├── base_speakers   
      │   └── ses   
      │       ├── en-au.pth   
      │       ├── en-br.pth   
      │       ├── en-default.pth   
      │       ├── en-india.pth   
      │       ├── en-newest.pth   
      │       ├── en-us.pth   
      │       ├── es.pth   
      │       ├── fr.pth   
      │       ├── jp.pth   
      │       ├── kr.pth   
      │       └── zh.pth   
      ├── converter   
      │   ├── checkpoint.pth   
      │   └── config.json   
      └── myshell-ai   
          ├── MeloTTS-Chinese   
          │   ├── checkpoint.pth   
          │   ├── config.json   
          │   └── README.md   
          └── MeloTTS-English-v2   
              ├── checkpoint.pth   
              ├── config.json   
              └── README.m   
      ```   

5. 下载`silero-vad`到`download/torch_hub_local/hub`目录
   ```bash
   git clone --depth 1  -b v3.0 https://github.com/snakers4/silero-vad download/torch_hub_local/hub/snakers4_silero-vad_v3.0
   ```
6. （可选）网络不好的童鞋，可以提前下载`nltk_data`到download目录。
   - 百度网盘链接：https://pan.baidu.com/s/1WKUtCMcqyxATRy6PQsFoUg?pwd=urbc 提取码：urbc 
   - 123网盘链接：https://www.123865.com/s/oEqDVv-SxBo?提取码:CnAJ

7. 将[bert-base-multilingual-uncased](https://huggingface.co/google-bert/bert-base-multilingual-uncased)和[bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)放到本项目根目录下的`download`文件夹。
   - 百度网盘链接：https://pan.baidu.com/s/1T9LKFeCgWssBTda3zz9m_g?pwd=6px4  提取码：6px4 


8. 运行`open_voice_v2/convert_to_ov.py`，将pytorch代码转成openvino专用格式。
   ```bash
   python open_voice_v2/convert_to_ov.py
   ```
9. 对于Win10/Win11系统，打开设置，搜索开发者设置，勾选`开发者模式`。不开启则下面运行api会报没有权限创建软链接。[参考链接](https://www.scivision.dev/windows-symbolic-link-permission-enable/)
![development_mode](../images/development_mode.png)

10. 输入下面的命令，运行api
   ```bash
   cd open_voice_v2
   uvicorn api:app --host 127.0.0.1  --port 5059 --workers 1
   ```
   