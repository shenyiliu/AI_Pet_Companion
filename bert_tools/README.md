### 使用方法
1. 将数据集丢到`data_dir`文件夹下
2. 将[预训练模型](https://hf-mirror.com/google-bert/bert-base-chinese)丢到`pretrain_models`文件夹,例如`bert_tools/pretrain_models/bert-base-chinese`
3. 复用主项目的openvino虚拟环境
4. 将`train.py`的main函数的is_train参数改成True(训练的时候用)

### 文件说明
1. `train.py`训练模型用
2. `config.yaml` 配置文件信息
3. `my_config.py`用于导入配置文件，并且包含部分配置文件
4. `my_tools`文件夹，包括数据导入，模型等信息。
5. `out_model`输出训练好的模型，用该模型输出预测结果
6. 验证日志
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
- FastTransFormer: [链接](https://github.com/NVIDIA/FasterTransformer)
