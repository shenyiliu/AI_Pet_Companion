import sys
import os
import torch
import torch.nn as nn
from collections import OrderedDict
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(os.path.dirname(dir_path))
from my_tools.model2 import Embedding, process_weights
from my_tools.encoder import EncoderWeights, CustomEncoder
from my_config import MyConfig


my_params = MyConfig()


class AfterModel(torch.nn.Module):
    """
    数据后处理
    """
    def __init__(
            self, weights, hidden_size: int = 768, num_entities: int = 20,
            num_intents: int = 5, hiden_drop_prob: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(hiden_drop_prob)
        self.start_entities_classifier = torch.nn.Linear(
            hidden_size, num_entities
        )
        self.end_entities_classifier = torch.nn.Linear(
            hidden_size, num_entities
        )
        self.intents_classifier = torch.nn.Linear(
            hidden_size, num_intents
        )
        # self.dense = torch.nn.Linear(hidden_size, hidden_size)
        # self.active = torch.nn.Tanh()
        w = OrderedDict()
        for k, v in weights.weights.items():
            if "entities" in k or "intent" in k and not k.endswith("_amax"):
                w[k] = v
            # if k.startswith("bert.pooler") and not k.endswith("_amax"):
            #     w[k[12:]] = v
        self.load_state_dict(w)

    def forward(self, outputs):
        sequence_output = self.dropout(outputs)
        # 取第一层做意图分类用
        pooled_output = outputs[:, :1]
        pooled_output = self.dropout(pooled_output)
        start_entities_logits = self.start_entities_classifier(sequence_output)
        end_entities_logits = self.end_entities_classifier(sequence_output)
        intent_logits = self.intents_classifier(pooled_output)
        # return dict(
        #     start_entity=start_entities_logits,
        #     end_entity=end_entities_logits,
        #     intent=intent_logits
        # )
        return (start_entities_logits, end_entities_logits, intent_logits)


class DIETClassifier(nn.Module):
    def __init__(self, config, weights, sparse=False, use_fp16=False):
        super().__init__()
        weights = process_weights(weights)
        self.embeddings = Embedding(config, weights)
        weights = EncoderWeights(
            layer_num=config.num_hidden_layers,
            hidden_dim=config.hidden_size,
            weights=weights,
            sparse=sparse
            )
        self.use_fp16 = use_fp16
        if use_fp16:
            weights.to_half()
        
        so_path = my_params.so_file_path
        self.encoder = CustomEncoder(
            layer_num=config.num_hidden_layers,
            head_num=config.num_attention_heads,
            head_size=int(config.hidden_size/config.num_attention_heads),
            weights=weights,
            remove_padding=True,
            sparse=sparse,
            path=so_path
        )
        self.after_model = AfterModel(weights)
        if use_fp16:
            self.after_model.half()
            # 注意encode不用half，因为它是c++写的，会自动重载half计算
         
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        前向计算，省略模型训练过程，因为c++部分的so暂时不确定能否训练，所以只做预测即可
        """
        # embedding计算
        embedding_states = self.embeddings(input_ids, token_type_ids)
        if self.use_fp16:
            embedding_states = embedding_states.half()

        # encoder计算
        max_seq_len = attention_mask.shape[1]
        attention_mask1 = attention_mask.view(-1, 1, 1, max_seq_len)
        attention_mask2 = attention_mask1.transpose(2, 3)
        attention_mask3 = attention_mask1 * attention_mask2
        attention_mask4 = attention_mask.unsqueeze(-1)
        mem_seq_lens2 = attention_mask.sum(axis=1, dtype=torch.int32)
        encoder_outputs = self.encoder(
            embedding_states, attention_mask3, mem_seq_lens2
        )[0] * attention_mask4
        # 解决mask值为0时fp16溢出问题
        encoder_outputs = torch.where(
            torch.isnan(encoder_outputs) * torch.eq(attention_mask4, 0),
            torch.full_like(encoder_outputs, 0),
            encoder_outputs
        )
        outputs = self.after_model(encoder_outputs)
        # after后向计算
        return {"logits": outputs}
