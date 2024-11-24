### 使用方法
1. 将数据集丢到`data_dir`文件夹下
2. 将[预训练模型](https://hf-mirror.com/google-bert/bert-base-chinese)丢到`pretrain_models`文件夹,例如`bert_tools/pretrain_models/bert-base-chinese`
3. 复用主项目的openvino虚拟环境
4. (可选)执行train.py训练模型
   ```bash
   cd bert_tools 
   python train.py
   ```

5. 将训练好的模型文件best.pth，放到当前目录的output_model文件夹
   - 训练好的模型：通过百度网盘分享的文件：链接：https://pan.baidu.com/s/1OnpsSNmQ-CA5_VF8CdC-ig?pwd=mxj7 提取码：mxj7

6. 运行convert_to_ov.py，将pytorch模型转openvino模型。
   ```bash
   python convert_to_ov.py
   ```
7. 测试效果。
   ```bash
   python predict.py
   ```
   
8. 部署api
   ```bash
   uvicorn api:app --host 127.0.0.1  --port 5518 --workers 1
   ```

### 文件说明
1. `train.py`训练模型用
2. `config.yaml` 配置文件信息
3. `my_config.py`用于导入配置文件，并且包含部分配置文件
4. `my_tools`文件夹，包括数据导入，模型等信息。
5. `out_model`输出训练好的模型，用该模型输出预测结果
6. `predict.py`执行推理用
7. `api.py` 部署api用

### 训练日志
```bash
valid metrics info:
AIRPLANE_MODE/On: acc: 1.00000  recall: 1.00000 f1: 1.00000     number: 15
BRIGHTNESS/To: acc: 1.00000     recall: 1.00000 f1: 1.00000     number: 13
BRIGHTNESS/Add: acc: 1.00000    recall: 1.00000 f1: 1.00000     number: 3
BRIGHTNESS/Sub: acc: 1.00000    recall: 1.00000 f1: 1.00000     number: 4
CALCULATOR/Off: acc: 1.00000    recall: 1.00000 f1: 1.00000     number: 17
CALCULATOR/On: acc: 1.00000     recall: 1.00000 f1: 1.00000     number: 18
POWER_SAVING_MODE/On: acc: 0.83333      recall: 0.90909 f1: 0.86957     number: 11
POWER_SAVING_MODE/Off: acc: 1.00000     recall: 1.00000 f1: 1.00000     number: 11
TASK_MANAGER/On: acc: 1.00000   recall: 1.00000 f1: 1.00000     number: 5
TASK_MANAGER/Off: acc: 1.00000  recall: 1.00000 f1: 1.00000     number: 5
VIDEO_CHAT/Close: acc: 1.00000  recall: 1.00000 f1: 1.00000     number: 5
VIDEO_CHAT/Open: acc: 1.00000   recall: 1.00000 f1: 1.00000     number: 6
VOLUME/To: acc: 1.00000 recall: 0.93333 f1: 0.96552     number: 15
VOLUME/Close: acc: 1.00000      recall: 1.00000 f1: 1.00000     number: 6
VOLUME/Add: acc: 1.00000        recall: 1.00000 f1: 1.00000     number: 3
VOLUME/Sub: acc: 0.83333        recall: 0.83333 f1: 0.83333     number: 6
overall_recall: 0.97902
overall_precision: 0.97902
overall_f1: 0.97902
intent acc : 100.00%
```

### 注：
- 参考论文：[DIET,  Dual Intent and Entity Transformer](https://arxiv.org/pdf/2004.09936.pdf)
