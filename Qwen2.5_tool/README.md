# 处理Qwen2.5 7B模型工具包

## 1. 安装依赖

直接复用根目录的环境
```bash
pip install -r requirements.txt
```

## 2. 下载模型
在魔搭社区中下载Qwen2.5-7B-Instruct模型，并保存在download目录下

下载链接：https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct

## 3. 转换模型并量化
在根目录中运行下述转换命令
```bash
optimum-cli export openvino --model download/Qwen2.5-7B-Instruct --weight-format int4 --task text-generation-with-past output/ov-qwen2.5-7b-instruct-int4
```
