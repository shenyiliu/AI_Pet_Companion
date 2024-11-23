import os
import json
from yaml import safe_load
from colored import bg, stylize


class MyConfig:
    """
    我的配置文件
    """
    def __init__(self):
        self._now_dir = os.path.dirname(os.path.abspath(__file__))
        self._config_path = os.path.join(self._now_dir, "config.yaml")
        with open(self._config_path, "rt", encoding="utf-8") as f:
            self._yaml_dict = safe_load(f)
        # 储存文件夹信息
        self.dir_info = {}
        self.init_dir()
        # 储存数据集路径
        self.dataset_dir = None
        # 储存预处理模型路径
        self.pre_model_path = None
        self.init_file_path()
        self.model_info = {}
        self.init_model_info()
        # 关系映射 + 实体映射
        self.intent2id, self.id2intent = self.get_intent()
        self.entity2id, self.id2entity = self.get_intent("entity")
        self.entity_size = len(self.entity2id)
        self.intent_num = len(self.intent2id)
        # FastTransFormer编译的so文件所在路径,用绝对路径
        self.so_file_path = "/workspace/FasterTransformer/build/lib/libth_bert.so"
        

    def init_dir(self):
        """
        初始化文件夹
        :return:
        """
        for k, v in self._yaml_dict["dir_info"].items():
            temp_dir = os.path.join(self._now_dir, v)
            self.dir_info[k] = temp_dir
            # 如果文件夹不存在, 创建相关文件夹
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)

    def init_file_path(self):
        """
        初始化文件夹信息
        :return:
        """
        self.dataset_dir = os.path.join(
            self.dir_info["data_dir"], self._yaml_dict["dataset_name"])
        self.pre_model_path = os.path.join(
            self.dir_info["pre_model_dir"], self._yaml_dict["pre_model_name"]
        )
        assert os.path.exists(self.dataset_dir),\
            f"数据集路径: {self.dataset_dir} 不存在, 请放置对应数据集文件"
        if not os.path.exists(self.pre_model_path):
            print(stylize(f"警告!模型路径: {self.pre_model_path} 不存在, 请放置对应预训练模型", bg("yellow")))

    def init_model_info(self):
        """
        初始化模型配置
        :return:
        """
        for k, v in self._yaml_dict["model_info"].items():
            self.model_info[k] = v

    def get_intent(self, label_type="intent"):
        """
        获取意图识别分类
        :return:
        """
        # 读取意图与id的映射表
        file_path1 = os.path.join(self.dataset_dir, f"{label_type}_label.json")
        assert os.path.exists(file_path1)
        with open(file_path1, "rt") as f:
            json_data = json.load(f)
            intent2id = json_data[f"{label_type}2id"]
            id2intent = json_data[f"id2{label_type}"]
            id2intent = {int(key): value for (key, value) in id2intent.items()}
        return intent2id, id2intent


if __name__ == '__main__':
    my_config = MyConfig()
    print(my_config)




