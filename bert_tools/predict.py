import os.path
import re
import torch
from torch.utils.data import DataLoader, SequentialSampler
from train import evaluate
from transformers import BertConfig
from my_tools.utils import load_checkpoint
from my_tools.data_load import MyDataLoad, MyDataSet
# from my_tools.model3 import DIETClassifier
from my_tools.model import DIETClassifier
from my_config import MyConfig

params = MyConfig()


class Predict(MyDataLoad):
    def __init__(self, model_path):
        """
        初始化
        :param model_path: 训练后的模型路径
        """
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model(model_path)
        self.max_seq_len = params.model_info["max_seq_len"]
        # self.tokenizer = None

    def load_model(self, model_path):
        """
        加载模型
        :return:
        """
        print("加载模型中")
        if model_path.endswith(".tar"):
            self.model, _, _ = load_checkpoint(model_path)
        else:
            # json_path = os.path.join(params.pre_model_path, "config.json")
            # bert_config = BertConfig.from_json_file(json_file=json_path)
            # weights = torch.load(model_path, map_location=torch.device("cpu"))
            # for v2/v3
            # sparse = params.model_info["use_sparse"]
            # use_fp16 = params.model_info["use_fp16"]
            # self.model = DIETClassifier(bert_config, weights, sparse, use_fp16)
            # for v1
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            self.model = DIETClassifier.from_pretrained(params.pre_model_path)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print("加载模型完毕")

    def get_dataloader(self, text_list: list, batch_size=64):
        feature = {"text": text_list}
        features = self.convert_data_list("预测", feature, True)
        my_dataset = MyDataSet(features)
        data_sample = SequentialSampler(my_dataset)
        my_dataloader = DataLoader(
            my_dataset,
            sampler=data_sample,
            batch_size=batch_size,
        )
        return my_dataloader

    @staticmethod
    def get_number(str1: str):
        """
        提取字符串中的数字
        :param str1:
        :return:
        """
        try:
            if "." in str1:
                num = float(str1)
            else:
                num = int(str1)
            return num
        except Exception as err:
            p = re.compile("[0-9]+[\.]?[0-9]*")
            res1 = p.search(str1)
            if res1 is not None:
                str1 = res1.group()
                str1 = str1.lstrip(".")
                try:
                    if "." in str1:
                        num = float(str1)
                    else:
                        num = int(str1)
                    return num
                except Exception as err:
                    return None
            else:
                return None

    def predict(self, text_list, is_one_intent: bool = True,
                is_strip: bool = False, batch_size=64):
        """
        预测学术要求分类
        :param text_list: 文本信息，每段一个文本
        :param is_one_intent: 是否每个句子仅一个类别，如果选择否，则单个句子可以输出多个类别
        :param is_strip: 是否剔除不含实体的多余文本
        :param batch_size: 批处理数据量，理论上设置越大则速度越快
        :return:
        """
        my_dataloader = self.get_dataloader(text_list, batch_size=batch_size)
        label_result = evaluate(
            self.model,
            device1=self.device,
            valid_load1=my_dataloader,
            is_predict=True
        )
        print(label_result)
        # 再清理一次缓存
        torch.cuda.empty_cache()



if __name__ == '__main__':
    from pprint import pprint
    import json
    import time
    import numpy as np
    import random
    import timeit
    from pprint import pprint
    from train import evaluate
    now_dir = os.path.dirname(os.path.abspath(__file__))
    # old model v2
    # model_path1 = os.path.join(now_dir, "out_model", "best.pth.tar")
    # new old v3
    model_path1 = os.path.join(now_dir, "out_model", "best.pth")
    predict = Predict(model_path1)
    text_list1 = [
        "帮我打开任务管理器",
        "你好",
        "让我们开始AI视频聊天吧"
    ]
    et = time.time()
    result_data2 = predict.predict(text_list1, False, True)
    pprint(result_data2)
    st = time.time()
    print("during ", st - et)
