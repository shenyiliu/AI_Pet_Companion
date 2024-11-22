import os.path

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from transformers.models.bert.modeling_bert import BertEncoder, BertEmbeddings
from my_config import MyConfig


my_params = MyConfig()


class Embedding(BertEmbeddings):
    def __init__(self, config, weights2) -> None:
        super().__init__(config)
        # 加载自定义权重
        for i in range(2):
            if weights2.get("weights", None) is not None:
                weights2 = weights2.weights
        w = OrderedDict()
        for k, v in weights2.items():
            if k.startswith("bert.embeddings") and not k.endswith("_amax"):
                w[k[16:]] = v
            if k.startswith("embeddings") and not k.endswith("_amax"):
                w[k[11:]] = v
        if "position_ids" not in w:
            w["position_ids"] = torch.arange(
                config.max_position_embeddings,
                dtype=torch.long).unsqueeze(0)
        self.load_state_dict(w)


def process_weights(weights2):
    # 处理Weight
    for i in range(2):
        if weights2.get("weights", None) is not None:
            weights2 = weights2.weights
    weights3 = {}
    for k, v in weights2.items():
        ks = k.split('.')
        if ks[-2] == 'LayerNorm':
            if ks[-1] == 'gamma':
                ks[-1] = 'weight'
            elif ks[-1] == 'beta':
                ks[-1] = 'bias'
        weights3['.'.join(ks)] = v
    return weights3


class DIETClassifier(nn.Module):
    def __init__(self, config, weights2):
        """
        Create DIETClassifier model
        """
        super().__init__()
        self.entity2id = my_params.entity2id
        self.num_entities = my_params.entity_size
        self.intents_list = my_params.intent2id
        self.num_intents = my_params.intent_num
        # 预先处理一下Weights
        weights2 = process_weights(weights2)
        self.embeddings = Embedding(config=config, weights2=weights2)
        self.encoder = BertEncoder(config)
#
        w = {}
        for k, v in weights2.items():
            if k.startswith('bert.encoder') and not k.endswith('_amax'):
                w[k[13:]] = weights2[k]
        self.encoder.load_state_dict(w)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
        self.start_entities_classifier = nn.Linear(
            config.hidden_size, self.num_entities
        )
        self.end_entities_classifier = nn.Linear(
            config.hidden_size, self.num_entities
        )
        self.intents_classifier = nn.Linear(
            config.hidden_size, self.num_intents
        )
        # self.init_weights()
        self.head_mask = [None] * config.num_hidden_layers
        """
        # 调试代码用，训练的时候注释掉
        w2 = OrderedDict()
        for k, v in weights2.items():
            if k.startswith('bert.encoder') and not k.endswith('_amax'):
                w2[k[5:]] = weights2[k]
            if k.startswith('bert.embeddings') and not k.endswith('_amax'):
                w2[k[5:]] = weights2[k]
            if "entities" in k or "intent" in k and not k.endswith("_amax"):
                w2[k] = v
        self.load_state_dict(w2)
        """

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask=None,
            intent_id=None,
            start_entity_ids=None,
            end_entity_ids=None
    ):
        """
        training model if entities_labels and intent_labels are passed, else inference
#
        :param input_ids
        :param token_type_ids
        :param attention_mask: attention_mask
        :param intent_id: labels of intent
        :param start_entity_ids: labels of entities
        :param end_entity_ids: labels of entities
        :return:
        """
        hidden_states = self.embeddings(input_ids, token_type_ids)
        max_seq_len = attention_mask.shape[1]
        attention_mask1 = attention_mask.view(-1, 1, 1, max_seq_len)
        attention_mask2 = attention_mask1.transpose(2, 3)
        attention_mask3 = attention_mask1 * attention_mask2
        attention_mask4 = attention_mask.unsqueeze(-1)
        extended_attention_mask = (1.0 - attention_mask3) * (-10000.0)
        outputs = self.encoder(
            hidden_states, extended_attention_mask, self.head_mask,
            return_dict=False)
        # encode部分鉴定没有问题
        seq_output = outputs[0] * attention_mask4
        sequence_output = self.dropout(seq_output)
#
        pooled_output = seq_output[:, :1]  # 取最后一层作为分类用
        pooled_output = self.dropout(pooled_output)
#
        start_entities_logits = self.start_entities_classifier(sequence_output)
        end_entities_logits = self.end_entities_classifier(sequence_output)
        intent_logits = self.intents_classifier(pooled_output)
#
        entity_loss = None
        if start_entity_ids is not None and end_entity_ids is not None:
            entities_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_start_logits = start_entities_logits.view(
                    -1, self.num_entities
                )
                active_start_logits = active_start_logits[active_loss]
#
                active_end_logits = end_entities_logits.view(
                    -1, self.num_entities
                )
                active_end_logits = active_end_logits[active_loss]
#
                # active_start_labels = torch.where(
                #     active_loss, start_entity_ids.view(-1),
                #     torch.tensor(entities_loss_fct.ignore_index)\
                #         .long().to(attention_mask.device)
                # )
                # active_end_labels = torch.where(
                #     active_loss, end_entity_ids.view(-1),
                #     torch.tensor(entities_loss_fct.ignore_index)\
                #         .long().to(attention_mask.device)
                # )
                active_start_labels = start_entity_ids.view(-1)[active_loss]
                active_end_labels = end_entity_ids.view(-1)[active_loss]
                start_entities_loss = entities_loss_fct(
                    active_start_logits, active_start_labels
                )
                end_entities_loss = entities_loss_fct(
                    active_end_logits, active_end_labels
                )
            else:
                start_entities_loss = entities_loss_fct(
                    start_entities_logits.view(-1, self.num_entities),
                    start_entity_ids.view(-1)
                )
                end_entities_loss = entities_loss_fct(
                    end_entities_logits.view(-1, self.num_entities),
                    end_entity_ids.view(-1)
                )
            entity_loss = (start_entities_loss + end_entities_loss) / 2
#
        intent_loss = None
        if intent_id is not None:
            intent_loss_fct = CrossEntropyLoss()
            intent_loss = intent_loss_fct(
                intent_logits.view(-1, self.num_intents),
                intent_id.view(-1)
            )
#
        if (start_entity_ids is not None and end_entity_ids is not None) \
                and (intent_id is not None):
            loss = entity_loss * 0.8 + intent_loss * 0.2
        else:
            loss = None
#
        return dict(
            entity_loss=entity_loss,
            intent_loss=intent_loss,
            loss=loss,
            logits=(start_entities_logits, end_entities_logits, intent_logits)
        )


def compare(out_dict1, out_dict2):
    out_dict3 = {}
    for k, v in out_dict2.items():
        if isinstance(v, torch.Tensor):
            out_dict3[k] = v.cpu().data.numpy()
        else:
            out_dict3[k] = v
    max_list = []
    for k in out_dict1.keys():
        temp_max = np.max(abs(out_dict1[k] - out_dict3[k]))
        max_list.append(temp_max)

    print("\n", "=" * 20)
    print("compare diff max", max_list, max(max_list))
    print("=" * 20, "\n")


if __name__ == '__main__':
    from transformers import BertConfig
    import numpy as np
    import random
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    random.seed(2020)
    np.random.seed(2020)

    config_path = os.path.join(my_params.pre_model_path, "config.json")
    bert_config = BertConfig.from_json_file(config_path)
    pre_model_path = os.path.join(
        my_params.dir_info["out_model"], "best.pth")
    weights = torch.load(pre_model_path)
    weights = process_weights(weights)
    # 先对比embedding咯
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载输入数据
    input_path = os.path.join(my_params.dir_info["out_model"], "input.npz")
    input_data = np.load(input_path)
    input_ids1 = torch.from_numpy(input_data["input_ids"])
    token_type_ids1 = torch.from_numpy(input_data["token_type_ids"])
    attention_mask1 = torch.from_numpy(input_data["attention_mask"])
    input_ids1 = input_ids1.to(device=device)
    token_type_ids1 = token_type_ids1.to(device=device)
    attention_mask1 = attention_mask1.to(device=device)
    # 获取原始输出数据
    output_path = os.path.join(my_params.dir_info["out_model"], "output_raw.npz")
    torch.cuda.empty_cache()
    # 获取输出数据
    model = DIETClassifier(bert_config, weights)
    model.to(device)
    model.eval()

    start, end, intent = model(
        input_ids1, token_type_ids1, attention_mask1)["logits"]
    output_data2 = {
        "start_entity": start.cpu().data.numpy(),
        "end_entity": end.cpu().data.numpy(),
        "intent": intent.cpu().data.numpy()
    }

    raw_output_data = np.load(output_path)
    # 再次对比
    compare(raw_output_data, output_data2)



