"""
用于加载数据集数据
"""
import os
import json

import numpy as np
import torch
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from my_config import MyConfig


# 加载自定义配置
params = MyConfig()


class MyDataSet(Dataset):
    def __init__(self, data_features):
        super().__init__()
        self.data = data_features

    def __getitem__(self, index):
        item = dict(
            input_ids=self.data["input_ids"][index],
            token_type_ids=self.data["token_type_ids"][index],
            attention_mask=self.data["attention_mask"][index],
            offset_mapping=self.data["offset_mapping"][index],
            length=self.data["length"][index],
            text=self.data["text"][index]
        )
        if "start_entity_ids" in self.data.keys():
            item["start_entity_ids"] = self.data["start_entity_ids"][index]
        if "end_entity_ids" in self.data.keys():
            item["end_entity_ids"] = self.data["end_entity_ids"][index]
        if "intent_id" in self.data.keys():
            item["intent_id"] = self.data["intent_id"][index]
        return item

    def __len__(self):
        return len(self.data["text"])


class MyDataLoad(object):
    def __init__(self):
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
           params.pre_model_path
        )

    @staticmethod
    def load_json_data(dataset_type: str, data_list=None):
        """
        读取json格式的样本信息
        :param dataset_type: 数据集类型：train, valid, test
        :param data_list:
        :return:
        """
        features = {"text": [], "entity": [], "intent_id": []}
        file_path = os.path.join(
            params.dataset_dir, f"{dataset_type}.json")
        if data_list is None:
            assert os.path.exists(file_path), \
                f"{dataset_type}集文件不存在, 请检查{file_path}路径"
            with open(file_path, "rt", encoding="utf-8") as f:
                data_list = json.load(f)
        intent2id = params.intent2id
        entity2id = params.entity2id
        for data in data_list:
            text = data["text"]
            # 收集实体信息，用于后期训练实体识别模型
            entity_list = []
            intent_id = intent2id.get(data["intent"])
            for entity in data["entity"]:
                entity_list.append(
                    {
                        "entity_id": entity2id[entity[-1]],
                        "position": entity[:-1]
                    }
                )
            features["text"].append(text)
            features["intent_id"].append(intent_id)
            features["entity"].append(entity_list)
        return features

    @staticmethod
    def binary_search(entity: dict, offset_list, left: int, right: int,
                      is_greater: bool) -> dict:
        """
        利用二分搜索查找token对应的最近实体所在位置
        :param entity: 实体信息，dict类型，包含entity_id:int以及position:[]
        :param offset_list: token的偏移信息, 记录分词后的每个token的跨度
        :param left: 记录实体左边开始位置，一般初始值为1，第一个是[CLS]
        :param right: 记录实体右边结束位置，一般是1 + 句子长度，不考虑[PAD]
        :param is_greater: 是否找大于等于的数
        :return: 返回最近的目标实体
        """
        # 如果找大于等于的数，也就是找实体的起始点
        if is_greater:
            target = entity["position"][0]
            num_list = [offset[0] for offset in offset_list]
        else:
            target = entity["position"][1]
            num_list = [offset[1] for offset in offset_list]
        while left + 1 < right:
            mid = left + (right - left) // 2
            if num_list[mid] == target:
                return mid
            elif num_list[mid] < target:
                left = mid
            else:
                right = mid
        # 如果targe在中间
        if is_greater:
            if num_list[left] <= target < num_list[right]:
                return left
            else:
                # 否则返回最右边的值
                return right
        else:
            if num_list[left] < target <= num_list[right]:
                return right
            else:
                return left

    def get_entity_ids(self, offset_list, temp_entity_list, max_seq_len: int):
        start_entity_ids = np.zeros(max_seq_len)
        end_entity_ids = np.zeros(max_seq_len)
        # 计算temp_offer_list左右两边边界
        left = right = 1
        for offset in offset_list[left:]:
            if sum(offset) > 0:
                right += 1
            else:
                break
        length = right - left
        right -= 1
        for entity in temp_entity_list:
            start_idx = self.binary_search(
                entity, offset_list, left, right, is_greater=True
            )
            end_idx = self.binary_search(
                entity, offset_list, left, right, is_greater=False
            )
            start_entity_ids[start_idx] = entity["entity_id"]
            end_entity_ids[end_idx] = entity["entity_id"]
        return start_entity_ids, end_entity_ids, length

    def convert_data_list(self, dataset_type: str, features: dict,
                          is_predict: bool = False):
        """
        转化数据，从list变成tensor
        :param dataset_type: 数据集类型
        :param features: 从json文件中加载的数据
        :param is_predict: 是否为预测集
        :return:
        """
        max_seq_len = params.model_info["max_seq_len"]
        # 将text打包成token,并且截断，填充pad, 为了方便起见返回offset用于生成每个token的实体类型
        features.update(
            self.tokenizer(
                features["text"],
                return_tensors="pt",
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=max_seq_len
            )
        )
        if is_predict:
            # 计算temp_offer_list左右两边边界
            length_list = []
            for offset_list in features["offset_mapping"]:
                left = right = 1
                for offset in offset_list[left:]:
                    if sum(offset) > 0:
                        right += 1
                    else:
                        break
                length = right - left
                length_list.append(length)
            features["length"] = torch.tensor(length_list).long()
            return features
        # 将entity对应的position, entity_id转化为token对应的entity_id
        features["entity_ids"] = []
        data_iter = tqdm(range(len(features["text"])))
        start_entity_ids = np.zeros(
            (len(data_iter), max_seq_len), dtype=np.int64
        )
        end_entity_ids = np.zeros(
            (len(data_iter), max_seq_len), dtype=np.int64
        )
        length_list = np.zeros(len(data_iter))
        for idx in data_iter:
            data_iter.set_description(f"正在获取{dataset_type}数据集实体信息")
            # 用于收集当前句子的实体信息
            # entity_ids = []
            temp_entity_list = features["entity"][idx]
            # 改成双指针网络格式，防止B-I预测时前后乱序问题
            temp_offset_list = features["offset_mapping"][idx].data.numpy()
            temp_start_ids, temp_end_ids, length = self.get_entity_ids(
                temp_offset_list, temp_entity_list, max_seq_len)
            start_entity_ids[idx] = temp_start_ids
            end_entity_ids[idx] = temp_end_ids
            length_list[idx] = length
        features["start_entity_ids"] = torch.tensor(start_entity_ids).long()
        features["end_entity_ids"] = torch.tensor(end_entity_ids).long()
        features["length"] = torch.tensor(length_list).long()
        features["intent_id"] = torch.tensor(
            features["intent_id"], dtype=torch.long
        )
        return features

    def load(self, dataset_type):
        """
        加载数据
        :param dataset_type: 数据集类型
        :return:
        """
        assert dataset_type in ["train", "valid", "test"],\
            "当前仅支持train, valid, test三种数据集加载"
        # 数据缓存路径，防止重复加载
        cache_path = os.path.join(params.dataset_dir, f"{dataset_type}.cache")
        if os.path.exists(cache_path):
            print(f"正在加载旧的{dataset_type}数据集")
            start_t = time.time()
            features = torch.load(cache_path)
            end_t = time.time()
            print("加载旧{}数据集成功，用时{:.4f}秒".format(
                dataset_type, end_t - start_t)
            )
        else:
            # 加载数据集信息
            features = self.load_json_data(dataset_type)
            features = self.convert_data_list(dataset_type, features)
        # 将features转化为DataSet
        print(f"开始转化{dataset_type}数据类型", f"一共{len(features['intent_id'])}条")
        my_dataset = MyDataSet(features)
        # 然后再转成dataloader
        if dataset_type == "train":
            # 先来一次数据随机采样
            data_sample = RandomSampler(my_dataset)
            my_dataloader = DataLoader(
                my_dataset,
                sampler=data_sample,
                batch_size=params.model_info["train_batch_size"],
            )
        else:
            if dataset_type == "valid":
                batch_size = params.model_info["valid_batch_size"]
            else:
                batch_size = params.model_info["test_batch_size"]
            data_sample = SequentialSampler(my_dataset)
            my_dataloader = DataLoader(
                my_dataset,
                sampler=data_sample,
                batch_size=batch_size,
            )
        print(f"转化{dataset_type}数据类型完成")
        return my_dataloader


if __name__ == '__main__':
    data_loader = MyDataLoad()
    # 测试一下二分搜索效果
    offset1 = [37, 38]
    entity_list1 = [
        {"position": [23, 29]},
        {"position": [37, 40]},
    ]
    # print(data_loader.binary_search(entity_list1, offset1))
    my_dataloader1 = data_loader.load("train")
    # # valid_loader = data_loader.load("valid")
    # batch = next(iter(my_dataloader1))
    # print(batch[0].shape)
    # tokenizer_22 = BertTokenizer.from_pretrained(
    #     os.path.join(params.pre_model_path, "vocab.txt")
    # )
    # for k, v in tokenizer_22.special_tokens_map.items():
    #     print(k, "-->", tokenizer_22.convert_tokens_to_ids(v), "-->", v)







