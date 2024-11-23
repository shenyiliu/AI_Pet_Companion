import os.path
import re
import torch
from torch.utils.data import DataLoader, SequentialSampler
from my_tools.utils import load_checkpoint
from my_tools.data_load import MyDataLoad, MyDataSet
# from my_tools.model3 import DIETClassifier
from my_tools.model import DIETClassifier
from my_tools.utils import get_entity, get_entity_triple
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
        # 意图和函数的映射
        self.intent2task = {
            # 多参数
            "VOLUME": {
                "fun": "set_volume",
                "entity": {
                    "Add": {"action": "+", "value": None},
                    "Sub": {"action": "-", "value": None},
                    "To": {"action": None, "value": None},
                    "Close": {"action": None, "value": 0}
                }
            },
            "BRIGHTNESS": {
                "fun": "set_brightness",
                "entity": {
                    "Add": {"action": "+", "value": None},
                    "Sub": {"action": "-", "value": None},
                    "To": {"action": None, "value": None},
                }
            },
            # 无参数
            "BATTERY": {"fun": "check_battery_status"},
            "SCREENSHOT": {"fun": "capture_screen"},
            "SYSTEM_INFO": {"fun": "get_system_info"},
            # 单参数
            "POWER_SAVING_MODE": {
                "fun": "set_power_mode",
                "entity": {"On": True, "Open": True, "Off": False, "Close": False}
            },
            "AIRPLANE_MODE": {
                "fun": "set_airplane_mode",
                "entity": {"On": True, "Open": True, "Off": False, "Close": False}
            },
            "CALCULATOR": {
                "fun": "control_calculator",
                "entity": {"On": True, "Open": True, "Off": False, "Close": False}
            },
            "TASK_MANAGER": {
                "fun": "control_task_manager",
                "entity": {"On": True, "Open": True, "Off": False, "Close": False}
            },
            "VIDEO_CHAT": {
                "fun": "camera_to_vLLM",
                "entity": {"On": True, "Open": True, "Off": False, "Close": False}
            }
        }

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
        self.model.eval()
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
        p = re.compile("[1-9]+[0-9]*")
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
            except Exception:
                return None
    
    def evaluate(self, valid_load1, threshold: float=0.8):
        """
        评估模型
        :param valid_load1: 待验证的数据
        :param threshold: 阀值，大于该数值才认为是正样本
        :return:
        """
        # 真实值，预测值，总量
        entity_pred_list = []
        id2entity = params.id2entity
        # eval_metric = SpanEntityScore(id2entity)
        intent_pred_list = []  # 收集意图预测结果
        probability_pred_list = [] # 收集意图预测概率
        entity_pred_info_list = []  # 收集实体最终预测结果
        # 取消打印进度条显示
        for data in valid_load1:
            data = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()

            }
            with torch.no_grad():
                output_data = self.model(
                    input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"],
                    token_type_ids=data["token_type_ids"],
                )
                start_entity_logits, end_entity_logits, intent_logits = \
                    output_data["logits"]
                start_entity_pred = torch.argmax(start_entity_logits, dim=-1)
                end_entity_pred = torch.argmax(end_entity_logits, dim=-1)
                start_entity_pred = start_entity_pred.detach().cpu().numpy()
                end_entity_pred = end_entity_pred.detach().cpu().numpy()
                # 获取token实际长度
                length_list = data["length"].detach().cpu().numpy().tolist()

                # -- 获取实体信息 -- #
                for idx in range(start_entity_pred.shape[0]):
                    length = length_list[idx]
                    temp_start_entity_pred = start_entity_pred[idx][1: length + 1]
                    temp_end_entity_pred = end_entity_pred[idx][1: length + 1]

                    # 收集实体初步预测值
                    temp_entity_pred = get_entity_triple(
                        temp_start_entity_pred,
                        temp_end_entity_pred,
                    )
                    # 获取筛选后实体，也就是刨除Text之类的实体信息
                    if len(temp_entity_pred) > 0:
                        text = data["text"][idx]
                        offset_list = data["offset_mapping"][idx][1:].detach().cpu()\
                            .numpy().tolist()
                        temp_list = get_entity(
                            text, offset_list, temp_entity_pred, id2entity)
                        entity_pred_info_list.append(temp_list)
                    else:
                        entity_pred_info_list.append([])
                    entity_pred_list.append(temp_entity_pred)

                # -- 收集意图信息 -- #
                # 如果是训练或者验证集，直接argmax,取所有最高分
                # 如果用阀值的话，可能变成多分类问题, 暂时只考虑单分类即可
                intent_logits = torch.squeeze(intent_logits, dim=1)
                probabilities =  torch.softmax(intent_logits, dim=1)
                for i in range(intent_logits.size(0)):
                    probability_tensor = probabilities[i]
                    max_id = torch.argmax(probability_tensor).item()
                    probability = torch.max(probability_tensor).item()
                    if probability < threshold:
                        max_id = -100
                    intent_pred_list.append(max_id)
                    probability_pred_list.append(probability)
        return {
            "intent_pred": intent_pred_list,
            "probability_pred": probability_pred_list,
            "entity_pred": entity_pred_info_list,
        }
    
    def map_task(self, text: str, intent_name: str, entity_action: str):
        """
        任务映射
        Args:
            text: 原文
            intent_name (str): 意图名称
            entity_action (str): 实体动作, 有些无实体意图可能为空
        """
        # 对于双参数意图
        if intent_name in {"VOLUME", "BRIGHTNESS"}:
            exe_func = self.intent2task[intent_name]["fun"]
            entity_dict = self.intent2task[intent_name]["entity"]
            action_dict = entity_dict[entity_action]
            if entity_action != "Close":
                # 提取数据
                value = self.get_number(text)
                if value is None:
                    value = 10
                action_dict["value"] = value
            return {"func": exe_func, "args": action_dict}
        # 对于无参数意图
        elif intent_name in {"BATTERY", "SCREENSHOT", "SYSTEM_INFO"}:
            exe_func = self.intent2task[intent_name]["fun"]
            return {"func": exe_func, "args": {}}
        # 对于单参数意图
        elif intent_name in intent2task:
            exe_func = self.intent2task[intent_name]["fun"]
            entity_dict = self.intent2task[intent_name]["entity"]
            value = entity_dict[entity_action]
            return {"func": exe_func, "args": {"action": None, "value": value}}
        # 对于无意图
        else:
            return {"func": None, "args": {}}

    def predict(self, text_list: list, threshold: float = 0.8, batch_size=64):
        """
        预测学术要求分类
        :param text_list: 文本信息，每段一个文本
        :param threshold: 阀值，大于该数值才认为是正样本
        :param batch_size: 批处理数据量，理论上设置越大则速度越快
        :return:
        """
        my_dataloader = self.get_dataloader(text_list, batch_size=min(batch_size, len(text_list)))
        label_result = self.evaluate(
            my_dataloader,
            threshold=threshold
        )
        intent_list = label_result["intent_pred"]
        entity_list = label_result["entity_pred"]
        entity_prob_list = label_result["probability_pred"]
        result_list = []
        for (intent_id, temp_entity_list, text, probability) in zip(
            intent_list, entity_list, text_list, entity_prob_list
        ):
            message = ""
            intent_name = params.id2intent.get(intent_id, None)
            if len(temp_entity_list) > 0:
                entity_dict = temp_entity_list[0]
                entity_type = entity_dict["entity_type"]
                (raw_intent_type, entity_action) = entity_type.split("/")
                # 必须一致，才能进行下一步
                if raw_intent_type == intent_name:
                    # 开始映射任务
                    data = self.map_task(text, intent_name, entity_action)
                else:
                    data = {}
            else:
                # 如果是无参数的意图
                if intent_name in {"BATTERY", "SCREENSHOT", "SYSTEM_INFO", None}:
                    data = self.map_task("", intent_name, "")  
                # 如果是双参数的意图，可以进一步询问
                elif intent_name in {"VOLUME", "BRIGHTNESS"}:
                    value = self.get_number(text)
                    if value is not None:
                        exe_func = self.intent2task[intent_name]["fun"]
                        data = {"func": exe_func, "args": {"action": None, "value": value}}
                    elif "最大" in text or "最高" in text:
                        data = {"func": "set_volume", "args": {"action": None, "value": 100}}
                    elif "最小" in text or "最低" in text:
                        data = {"func": "set_volume", "args": {"action": None, "value": 0}}
                    elif "大" in text:
                        data = {"func": "set_volume", "args": {"action": "+", "value": 10}}
                    elif "小" in text:
                        data = {"func": "set_volume", "args": {"action": "-", "value": 10}}
                    else:
                        message = "请问您要调到多大呢？"
                        data = {}
                # 如果是单参数的意图
                else:
                    message = "请问您是要打开还是关闭呢？"
                    data = {}
            temp_dict = {"data": data, "message": message, "probability": round(probability, 4)}
            result_list.append(temp_dict)
        return result_list


if __name__ == '__main__':
    from pprint import pprint
    import time
    now_dir = os.path.dirname(os.path.abspath(__file__))
    # old model v2
    # model_path1 = os.path.join(now_dir, "out_model", "best.pth.tar")
    # new old v3
    model_path1 = os.path.join(now_dir, "out_model", "best.pth")
    predict = Predict(model_path1)
    text_list1 = [
        "音量设置为50",
    ]
    et = time.time()
    result_data2 = predict.predict(text_list1)
    pprint(result_data2)
    st = time.time()
    print("during ", st - et)
