from collections import Counter


# 实体级评测方式
class SpanEntityScore(object):
    def __init__(self, id2entity: dict):
        self.id2entity = id2entity
        self.trues = []
        self.predicts = []
        self.rights = []

    @staticmethod
    def compute(true_num: int, predict_num: int, right):
        """
        用于计算得分
        :param true_num: 真实值数量
        :param predict_num: 预测值数量
        :param right: 预测正确的数
        :return:
        """
        recall = 0 if true_num == 0 else (right / true_num)
        precision = 0 if predict_num == 0 else (right / predict_num)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)
        return round(recall, 6), round(precision, 6), round(f1, 6)

    def get_result(self):
        score_result = {}
        true_counter = Counter([x[0] for x in self.trues])
        predict_counter = Counter([x[0] for x in self.predicts])
        right_counter = Counter([x[0] for x in self.rights])
        for entity_id, true_num in true_counter.items():
            entity_type = self.id2entity[str(entity_id)]
            predict_num = predict_counter.get(entity_id, 0)
            right_num = right_counter.get(entity_id, 0)
            recall, precision, f1 = self.compute(
                true_num, predict_num, right_num)
            score_result[entity_type] = {
                "acc": precision,
                'recall': recall,
                'f1': f1,
                "number": true_num
            }
        overall_true_num = len(self.trues)
        overall_pred_num = len(self.predicts)
        overall_right_num = len(self.rights)
        overall_recall, overall_precision, overall_f1 = self.compute(
            overall_true_num, overall_pred_num, overall_right_num
        )
        score_result["overall_recall"] = overall_recall
        score_result["overall_precision"] = overall_precision
        score_result["overall_f1"] = overall_f1
        return score_result

    def update(self, true_subject, pred_subject):
        self.trues.extend(true_subject)
        self.predicts.extend(pred_subject)
        self.rights.extend(
            [
                pre_entity for pre_entity in pred_subject
                if pre_entity in true_subject
            ]
        )