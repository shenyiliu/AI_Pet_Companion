# LLM OpenVINO

## 1.转换qwen2VL模型

### 环境

```Python
pip install -q "transformers>=4.45" "torch>=2.1" "torchvision" "qwen-vl-utils" "Pillow" "gradio>=4.36" --extra-index-url https://download.pytorch.org/whl/cpu
pip install -qU "openvino>=2024.4.0" "nncf>=2.13.0"
# pip install ipywidgets
```


## 转换模型

执行convert.py脚本，运行的时候会从huggingface下载模型，如果下载失败，请手动下载模型，并放置在download/Qwen2-VL-2B-Instruct目录下

## 运行测试程序

执行main.py脚本


