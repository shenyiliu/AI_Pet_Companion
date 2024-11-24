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

5. （可选）下载[WavMark](https://huggingface.co/M4869/WavMark)模型到download目录。
6. 下载`silero-vad`到`download/torch_hub_local/hub`目录
   ```bash
   git clone --depth 1  -b v3.0 https://github.com/snakers4/silero-vad download/torch_hub_local/hub/snakers4_silero-vad_v3.0
   ```
7. 运行`open_voice_v2/get_openvoice.py`获取OpenVoice代码，并进行少量修改。
   ```bash
   python open_voice_v2/get_openvoice.py 
   ```
   
8. （可选）修改`OpenVoice/openvoice/api`大概107行，修改后可以免去一个在线文件下载。
   - 修改前
   ```python
   ...
   import wavmark
   self.watermark_model = wavmark.load_model().to(self.device) 
   ```
   - 修改后
   ```python
   ...
   import wavmark
   self.watermark_model = wavmark.load_model(path="xxxxx/download/WavMark/step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl").to(self.device)  
   ```

9. 运行`open_voice_v2/convert_to_ov.py`，将pytorch代码转成openvino专用格式。
   ```bash
   python open_voice_v2/convert_to_ov.py
   ```
   
10. 对于Win10/Win11系统，打开设置，搜索开发者设置，勾选`开发者模式`。不开启则下面运行api会报没有权限创建软链接。[参考链接](https://www.scivision.dev/windows-symbolic-link-permission-enable/)
![development_mode](../images/development_mode.png)

11. 输入下面的命令，运行api（该步骤会从huggingface下载一些bert模型，注意你的网络）
   ```bash
   cd open_voice_v2
   uvicorn api:app --host 127.0.0.1  --port 5059 --workers 1
   ```
   