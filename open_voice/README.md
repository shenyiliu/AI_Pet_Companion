### 简单说明
- 参考了[官方教程](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvoice/openvoice.ipynb)
1. 复用项目根目录readme创建的openvino虚拟环境。
2. 另行安装一下ffmpeg
    ```bash
    conda install ffmpeg
    ```
3. 下载模型到download目录。[OpenVoice](https://huggingface.co/myshell-ai/OpenVoice), [WavMark](https://huggingface.co/M4869/WavMark)（可选）
4. 下载`silero-vad`到download目录
   ```bash
   git clone --depth 1  -b v3.0 https://github.com/snakers4/silero-vad download/snakers4-silero-vad-v3.0
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

7. 