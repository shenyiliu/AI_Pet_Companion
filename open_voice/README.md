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

3. 下载模型到download目录。[OpenVoice](https://huggingface.co/myshell-ai/OpenVoice), [WavMark](https://huggingface.co/M4869/WavMark)（可选）
4. 下载`silero-vad`到`download/torch_hub_local/hub`目录
   ```bash
   git clone --depth 1  -b v3.0 https://github.com/snakers4/silero-vad download/torch_hub_local/hub/snakers4_silero-vad_v3.0
   ```
5. 运行`open_voice/get_openvoice.py`获取OpenVoice代码，并进行少量修改。
   ```bash
   python open_voice/get_openvoice.py 
   ```
6. （可选）修改`OpenVoice/openvoice/api`大概107行，修改后可以免去一个在线文件下载。
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

7. 运行`open_voice/convert_to_ov.py`，将pytorch代码转成openvino专用格式。
   ```bash
   python open_voice/convert_to_ov.py
   ```
8. 输入下面的命令，运行api
   ```bash
   cd open_voice
   uvicorn api:app --host 127.0.0.1  --port 5059 --workers 1
   ```
   