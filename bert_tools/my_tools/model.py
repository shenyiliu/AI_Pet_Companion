import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from my_config import MyConfig


my_params = MyConfig()


class DIETClassifier(BertPreTrainedModel):
    def __init__(self, config):
        """
        Create DIETClassifier model
        """
        super().__init__(config)
        self.entity2id = my_params.entity2id
        self.num_entities = my_params.entity_size
        self.intents_list = my_params.intent2id
        self.num_intents = my_params.intent_num

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.start_entities_classifier = nn.Linear(
            config.hidden_size, self.num_entities
        )
        self.end_entities_classifier = nn.Linear(
            config.hidden_size, self.num_entities
        )
        self.intents_classifier = nn.Linear(
            config.hidden_size, self.num_intents
        )
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            intent_id=None,
            start_entity_ids=None,
            end_entity_ids=None,
    ):
        """
        training model if entities_labels and intent_labels are passed, else inference

        :param input_ids: embedding ids of tokens
        :param attention_mask: attention_mask
        :param token_type_ids: token_type_ids
        :param intent_id: labels of intent
        :param start_entity_ids: labels of entities
        :param end_entity_ids: labels of entities
        :return:
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        pooled_output = outputs[0][:, :1]  # 取最后一层作为分类用
        pooled_output = self.dropout(pooled_output)

        start_entity_logits = self.start_entities_classifier(sequence_output)
        end_entity_logits = self.end_entities_classifier(sequence_output)
        intent_logits = self.intents_classifier(pooled_output)

        entity_loss = None
        if start_entity_ids is not None and end_entity_ids is not None:
            entities_loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_start_logits = start_entity_logits.view(
                    -1, self.num_entities
                )
                active_start_logits = active_start_logits[active_loss]

                active_end_logits = end_entity_logits.view(
                    -1, self.num_entities
                )
                active_end_logits = active_end_logits[active_loss]

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
                    start_entity_logits.view(-1, self.num_entities),
                    start_entity_ids.view(-1)
                )
                end_entities_loss = entities_loss_fct(
                    end_entity_logits.view(-1, self.num_entities),
                    end_entity_ids.view(-1)
                )
            entity_loss = (start_entities_loss + end_entities_loss) / 2

        intent_loss = None
        if intent_id is not None:
            intent_loss_fct = CrossEntropyLoss()
            intent_loss = intent_loss_fct(
                intent_logits.view(-1, self.num_intents),
                intent_id.view(-1)
            )

        if (start_entity_ids is not None and end_entity_ids is not None) \
                and (intent_id is not None):
            loss = entity_loss * 0.8 + intent_loss * 0.2
        else:
            loss = None
        # train
        if entity_loss is not None and intent_loss is not None and loss is not None:
            return (entity_loss, intent_loss, loss)
        # predict
        else:
            return (
                start_entity_logits,
                end_entity_logits,
                intent_logits
            )