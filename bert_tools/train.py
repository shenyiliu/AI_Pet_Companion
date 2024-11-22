import torch
import random
import os
import logging
import numpy as np
from transformers import BertConfig
from tqdm import tqdm
from my_tools.data_load import MyDataLoad
from my_tools.utils import set_logger, RunningAverage, \
    load_checkpoint, save_checkpoint, get_entity, get_entity_triple
from my_tools.optimization import BertAdam
from my_config import MyConfig
from my_tools.model import DIETClassifier
# from my_tools.model2 import DIETClassifier, Embedding
from my_tools.metric import SpanEntityScore

params = MyConfig()


def evaluate(model1, device1, valid_load1, is_predict=False):
    """
    评估模型
    :param model1:
    :param device1:
    :param valid_load1:
    :param is_predict: 是否为预测集
    :return:
    """
    model1.eval()
    # 真实值，预测值，总量
    correct_num, predict_num, total_num = 0, 0, 0
    entity_pred_list = []
    id2entity = params.id2entity
    eval_metric = SpanEntityScore(id2entity)
    intent_pred_list = []  # 收集意图预测结果
    entity_pred_info_list = []  # 收集实体最终预测结果
    # 取消打印进度条显示
    for data in valid_load1:
        data = {
            k: v.to(device1) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()

        }
        with torch.no_grad():
            output_data = model1(
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
                if is_predict and len(temp_entity_pred) > 0:
                    text = data["text"][idx]
                    offset_list = data["offset_mapping"][idx][1:].detach().cpu()\
                        .numpy().tolist()
                    temp_list = get_entity(
                        text, offset_list, temp_entity_pred, id2entity)
                    entity_pred_info_list.append(temp_list)
                else:
                    entity_pred_info_list.append([])
                entity_pred_list.append(temp_entity_pred)

                # 验证集专用
                if not is_predict:
                    t_start_entity_ids = data["start_entity_ids"][idx][1: 1 + length]
                    t_end_entity_ids = data["end_entity_ids"][idx][1: 1 + length]
                    t_start_entity_ids = t_start_entity_ids\
                        .detach().cpu().numpy()
                    t_end_entity_ids = t_end_entity_ids\
                        .detach().cpu().numpy()
                    temp_entity_label = get_entity_triple(
                        t_start_entity_ids,
                        t_end_entity_ids
                    )
                    eval_metric.update(
                        true_subject=temp_entity_label,
                        pred_subject=temp_entity_pred
                    )

            # -- 收集意图信息 -- #
            # 如果是训练或者验证集，直接argmax,取所有最高分
            # 如果是predict, 则取阀值以上为最终得分
            if not is_predict:
                intent_pred = torch.argmax(
                    torch.softmax(intent_logits, dim=-1), dim=-1
                ).squeeze()
                # 计算intent打分
                correct_num += torch.eq(
                    intent_pred,
                    data["intent_id"]
                ).detach().data.sum().item()
                total_num += intent_pred.size(0)
            else:
                # 如果用阀值的话，可能变成多分类问题, 暂时只考虑单分类即可
                intent_logits = torch.squeeze(intent_logits, dim=-1)
                intent_pred_one_hot = torch.where(
                    torch.sigmoid(intent_logits) > 0.7,
                    torch.ones(intent_logits.size(), device=device1),
                    torch.zeros(intent_logits.size(), device=device1)
                )
                intent_pred = torch.zeros(intent_logits.size(0), device=device1)
                # 用-100 标记该样本为负样本
                # 如果没有对应实体或者分数阀值较低，则认为是负样本，也就是非学术要求
                negative_data = torch.tensor([-100], device=device1)
                for idx, sample in enumerate(intent_pred_one_hot):
                    if 1 not in sample or len(entity_pred_list[idx]) == 0:
                        sample = negative_data
                    else:
                        sample = torch.argmax(intent_logits[idx])
                    intent_pred[idx] = sample
            intent_pred = intent_pred.detach().cpu().numpy()
            # 收集意图预测值
            intent_pred_list.extend(intent_pred.tolist())
    # 清空缓存
    torch.cuda.empty_cache()
    if not is_predict:
        entity_result = eval_metric.get_result()
        # 格式化实体训练日志
        metrics_str = ""
        for k, v in entity_result.items():
            metrics_str += f"{k}: "
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if k2 != "number" and v2 != 0:
                        metrics_str += f"{k2}: {v2:0.5f}\t"
                    else:
                        metrics_str += f"{k2}: {v2}\t"
            elif isinstance(v, (float, np.float32, np.float64)):
                metrics_str += f"{v:0.5f}\t"
            else:
                metrics_str += f"{v}\t"
            metrics_str += "\n"
        logging.info("valid metrics info:\n" + metrics_str.rstrip("\n"))
        logging.info(
            "intent acc : {:.2f}%\n".format(correct_num / total_num * 100))
        result = {
            "f1": entity_result["overall_f1"],
            "recall": entity_result["overall_recall"],
            "precision": entity_result["overall_precision"]
        }
        return result
    else:
        return {
            "intent_pred": intent_pred_list,
            "entity_pred": entity_pred_info_list,
            "f1": 0
        }


def train_and_evaluate(
        model1, device1, optimizer1, train_load1, valid_load1, start_epoch1=0):
    """
    训练并且评估模型
    :param model1: 模型
    :param device1: 硬件设备,CPU与cuda
    :param optimizer1:  优化器
    :param train_load1: 训练数据
    :param valid_load1: 验证数据
    :param start_epoch1: 开始的epoch
    :return:
    """
    n_gpu1 = params.n_gpu
    gradient_acc_steps = params.model_info["gradient_acc_steps"]
    min_epoch_num = params.model_info["min_epoch_num"]
    patience_num = params.model_info["patience_num"]
    patience_rate = params.model_info["patience_rate"]
    best_valid_f1 = 0.0  # 最佳f1
    patience_count = 0  # 统计未提示效果次数
    for epoch1 in range(start_epoch1, params.model_info["epoch"]):
        logging.info("Epoch {} / {}".format(
            epoch1 + 1, params.model_info["epoch"]
        ))

        # 正式训练
        data_iter = tqdm(train_load1, ascii=True)
        model1.train()
        loss_avg = RunningAverage()
        loss_entity_avg = RunningAverage()
        loss_intent_avg = RunningAverage()
        # # 临时加入，用于打断训练集
        # count = 0
        for step, data in enumerate(data_iter):
            input_ids = data["input_ids"].to(device1)
            token_type_ids = data["token_type_ids"].to(device1)
            attention_mask = data["attention_mask"].to(device1)
            start_entity_ids = data["start_entity_ids"].to(device1)
            end_entity_ids = data["end_entity_ids"].to(device1)
            intent_id = data["intent_id"].to(device1)
            output_dict = model1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_entity_ids=start_entity_ids,
                end_entity_ids=end_entity_ids,
                intent_id=intent_id,
            )
            loss = output_dict["loss"]
            entity_loss = output_dict["entity_loss"]
            intent_loss = output_dict["intent_loss"]
            if n_gpu1 > 0:
                loss = loss.mean()
            if gradient_acc_steps > 1:
                loss = loss / gradient_acc_steps
            loss.backward()
            # 间隔n次再更新梯度，节省显存
            if (step + 1) % gradient_acc_steps == 0:
                optimizer1.step()
                model1.zero_grad()
            loss_avg.update(loss.item() * gradient_acc_steps)
            loss_entity_avg.update(entity_loss.item())
            loss_intent_avg.update(intent_loss.item())
            data_iter.set_postfix(
                loss="{:05.4f}".format(loss_avg()),
                loss_entity="{:05.4f}".format(loss_entity_avg()),
                loss_intent="{:05.4f}".format(loss_intent_avg())
            )
            # if count > 24:
            #     break
            # count += 1
        msg = "loss: {:05.5f} ".format(loss_avg())
        msg += "loss_entity: {:05.5f} ".format(loss_entity_avg())
        msg += "loss_intent: {:05.5f} ".format(loss_intent_avg())
        logging.info(msg)
        # 预测模型
        valid_result = evaluate(model1, device1, valid_load1)
        valid_f1 = valid_result['f1']

        # 保留最佳模型
        if valid_f1 > best_valid_f1:
            logging.info("fount new best valid f1 score")
            # 保存模型
            model_to_save = model.module \
                if hasattr(model, 'module') else model
            optimizer_to_save = optimizer
            save_checkpoint(
                {
                    'epoch': epoch1 + 1,
                    'model': model_to_save,
                    'optim': optimizer_to_save
                },
                is_best=True,
                checkpoint=params.dir_info["out_model"]
            )
            if valid_f1 - best_valid_f1 < patience_rate:
                patience_count += 1
            else:
                logging.info("best valid f1 score is {}\n".format(valid_f1))
                patience_count = 0
            best_valid_f1 = valid_f1
        else:
            patience_count += 1

        # 提前结束训练
        if epoch1 > min_epoch_num and patience_count > patience_num:
            logging.info("Best val f1: {:.4f}".format(best_valid_f1))
            break


if __name__ == '__main__':
    # 加一个参数is_train来判断是否要训练
    is_train = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.model_info["multi_gpu"]:
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
        else:
            n_gpu = 0
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(params.model_info["device_id"])
            n_gpu = 1
        else:
            n_gpu = 0
    params.device = device
    params.n_gpu = n_gpu
    # 设置随机种子
    seed = params.model_info["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # 设置日志
    log_path = os.path.join(params.dir_info["out_log"], "train.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    set_logger(save=True, log_path=log_path)
    logging.info("device: {}".format(device))
    logging.info("加载模型中")

    # 加载模型
    config_path = os.path.join(params.pre_model_path, "config.json")
    model_path = os.path.join(params.pre_model_path, "pytorch_model.bin")
    bert_config = BertConfig.from_json_file(config_path)
    # model = BertForRE(bert_config)
    # train for v1
    model = DIETClassifier.from_pretrained(
        config=bert_config,
        pretrained_model_name_or_path=params.pre_model_path
    )
    # for v2
    # model = DIETClassifier(bert_config, torch.load(model_path))
    logging.info("模型加载完毕")

    # 加载数据集
    my_data = MyDataLoad()
    if is_train:
        train_loader = my_data.load("train")
        valid_loader = my_data.load("valid")
        test_loader = []
    else:
        train_loader = []
        valid_loader = []
        test_loader = my_data.load("test")

    # -- 增加模型泛化能力 -- #
    # fine-tuning
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if
                        'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    # 参数decay
    optimizer_grouped_parameters = [
        # pretrain model param, 预训练模型参数
        {
            'params': [
                p for n, p in param_pre if
                not any(nd in n for nd in no_decay)
            ],
            'weight_decay': params.model_info["weight_decay_rate"],
            'lr': params.model_info["fine_tuning_lr"]
        },
        {
            'params': [
                p for n, p in param_pre if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
            'lr': params.model_info["fine_tuning_lr"]
        },
        # downstream model
        {
            'params': [
                p for n, p in param_downstream
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': params.model_info["weight_decay_rate"],
            'lr': params.model_info["down_entity_lr"]
        },
        {
            'params': [
                p for n, p in param_downstream
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0,
            'lr': params.model_info["down_entity_lr"]
        }
    ]
    # # 获取总训练次数
    if is_train:
        temp_a = params.model_info["gradient_acc_steps"]
        epoch = params.model_info["epoch"]
        num_train_optimization_steps = len(train_loader) // temp_a * epoch

        # 设置优化器
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            warmup=params.model_info["warmup_prob"],
            schedule="warmup_cosine",
            t_total=num_train_optimization_steps,
            max_grad_norm=params.model_info["clip_grad"]
        )
    else:
        optimizer = None
    # 加载历史模型记录(暂时只需要验证集加载就行了）
    if not is_train:
        restore_file_path = os.path.join(
            params.dir_info["out_model"], "best.pth.tar")
        if restore_file_path is not None:
            model, optimizer, start_epoch = load_checkpoint(restore_file_path)

    model.to(device)

    # 增加并行化训练支持
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # 正式训练
    if is_train:
        train_and_evaluate(model, device, optimizer, train_loader, valid_loader)
    else:
        # 直接验证
        evaluate(model, device, test_loader)
