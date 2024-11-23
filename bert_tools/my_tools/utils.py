import os
import torch
import shutil
import logging


class RunningAverage(object):
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
        :param log_path: 日志地址
        :param save: 是否保存，如果False则直接输出
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
            )
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains the entire model,
            may contain other keys such as epoch, optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        torch.save(
            state["model"].state_dict(), os.path.join(checkpoint, "best.pth")
        )


def load_checkpoint(checkpoint, optimizer=True):
    """Loads entire model from file_path. If optimizer is True, loads
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        optimizer: (bool) resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim'], checkpoint["epoch"]
    return checkpoint['model'], checkpoint["epoch"]


def span2str(triples, tokens):
    """
    将拆分后的span重新变成str
    :param triples:
    :param tokens:
    :return:
    """
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output


def entity_id2tag(entity_id_list: list, id2entity: dict) -> list:
    """
    用于将实体转化为对应的B-Entity与I-Entity模式的数据
    :param entity_id_list:
    :param id2entity:
    :return:
    """
    before_id = None
    entity_tag_list = []
    object_tag = id2entity[str(0)]
    for entity_id in entity_id_list:
        if entity_id == 0:
            entity_tag_list.append(object_tag)
        else:
            if before_id is not None:
                if before_id == entity_id:
                    entity_tag_list.append("I-" + id2entity[str(entity_id)])
                else:
                    entity_tag_list.append("B-" + id2entity[str(entity_id)])
            else:
                entity_tag_list.append("B-" + id2entity[str(entity_id)])
        before_id = entity_id
    return entity_tag_list


# def get_entity_triple(entity_pred, attention_mask):
#     """
#     获取实体三元组
#     :param entity_pred:
#     :param attention_mask
#     :return: [[start, end, entity_type]]
#     """
#     result_entity_list = []
#     # 收集当前所有实体
#     for entity_list, attention_list in zip(entity_pred, attention_mask):
#         # 双指针识别实体
#         i = 0
#         temp_entity_list = []
#         for j in range(len(entity_list)):
#             if not attention_list[j]:
#                 break
#             if entity_list[i] != entity_list[j]:
#                 if entity_list[i] > 0:
#                     temp_entity_list.append([i, j, entity_list[i]])
#                 i = j
#         # add tail entity
#         if entity_list[i] > 0 and i < len(entity_list) - 1\
#                 and entity_list[i] == entity_list[len(entity_list) - 1]:
#             temp_entity_list.append([i, len(entity_list), entity_list[i]])
#         result_entity_list.append(temp_entity_list)
#     return result_entity_list


def get_entity_triple(start_entity_ids, end_entity_ids):
    """
    获取实体三元组，根据开始实体与结束实体的位置，获取相邻且相等实体
    :param start_entity_ids:
    :param end_entity_ids:
    :return:
    """
    i = 0
    j = 0
    length = len(start_entity_ids)
    entity_list = []
    while j < length:
        if start_entity_ids[j] > 0:
            i = j
        if end_entity_ids[j] > 0:
            if end_entity_ids[j] == start_entity_ids[i]:
                entity_list.append([start_entity_ids[i], i, j])
            # 考虑到没有交叉实体问题，直接移动i
            i = j
        j += 1
    return entity_list


def get_entity(text: str, offset_list: list, entity_triples: list,
               id2entity: dict):
    """
    获取对应实体信息, 返回[dict{entity_type: "", position: {}}]
    :param text: 具体文本信息
    :param offset_list: text分割成token后，记录每个token的信息
    :param entity_triples: 包含[entity_id, start_id, end_id]的三元组列表
    :param id2entity: 将entity_id转化为对应的label
    :return:
    """
    result_list = []
    if len(entity_triples) > 0:
        for entity in entity_triples:
            start = offset_list[entity[1]][0]
            end = offset_list[entity[-1]][1]
            if start > 0 and end == 0:
                end = len(text)
            desc = text[start: end].rstrip()
            end = start + len(desc)
            entity_type = id2entity[int(entity[0])]
            temp_dict = {
                "position": [int(start), int(end)],
                "entity_type": entity_type,
                "description": desc
            }
            result_list.append(temp_dict)
    return result_list


if __name__ == '__main__':
    start_str = "0 11  0  0  0  0  0  0  9  0  0  0  0  0  0  0"
    start_list = [int(start) for start in start_str.split()]
    end_str = "0  0  0  0  0  0 11  0  9  0  0  0  0  0  0  0"
    end_list = [int(start) for start in end_str.split()]
    res = get_entity_triple(start_list, end_list)
    from pprint import pprint
    pprint(res)




