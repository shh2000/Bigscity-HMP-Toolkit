from torch.utils.data import DataLoader
from data_extract.route_planning_evaluate import route_planning_evaluate_data as rped
from _tkinter import _flatten
import json


class RoutePlanningEvaluate(object):
    def __init__(self, data, mode=1, k=5):
        # 加载json类型的数据为字典类型
        data_dict = json.loads(data)
        # 获得模型预测得到的位置数量
        self.len_pred = data_dict['len_pred']
        # 保留输入数据的用户位置信息 (以json字典对形式)
        del data_dict['len_pred']
        self.data = data_dict
        self.mode = mode
        self.k = k

    # traj1 traj2 输入两个轨迹，输出编辑距离
    def editDist(self, traj1, traj2):
        matrix = [[0 for j in range(len(traj2) + 1)] for i in range(len(traj1) + 1)]
        # print(matrix)
        for i in range(len(traj1) + 1):
            matrix[i][0] = i
        for j in range(len(traj2) + 1):
            matrix[0][j] = j
        for i in range(1, len(traj1) + 1):
            for j in range(1, len(traj2) + 1):
                if traj1[i-1] == traj2[j-1]:
                    matrix[i][j] = matrix[i - 1][j - 1]
                else:
                    matrix[i][j] = min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]) + 1
        return matrix[len(traj1)][len(traj2)]

    # traj1 traj2 输入两个轨迹，计算精确率 召回率 F1
    # traj1是原始数据 traj2是预测数据
    def calMetric(self, traj1, traj2):
        trajset1 = set(traj1[1:-1])  # 起点和终点不算进来
        trajset2 = set(traj2[1:-1])
        unionset = trajset1 & trajset2
        try:
            precision = len(unionset) / len(trajset2)
            recall = len(unionset) / len(trajset1)
            F1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            precision = 0
            recall = 0
            F1 = 0
        return precision, recall, F1

    # 统计所有的类别
    def get_unique_labels(self, y_true, y_pred):
        y_true_set = set(y_true)
        y_pred_set = set(_flatten(y_pred))
        unique_label_set = y_true_set | y_pred_set
        unique_label = list(unique_label_set)
        return unique_label

    # y_true: 1d-list-like
    # y_pred: 2d-list-like
    # k：针对top-k各结果进行计算（k <= y_pred.shape[1]）
    def precision_recall_fscore_k(self, y_trues, y_preds, k=3, digs=2):
        # 取每个样本的top-k个预测结果！
        y_preds = [pred[:k] for pred in y_preds]
        unique_labels = self.get_unique_labels(y_trues, y_preds)
        num_classes = len(unique_labels)
        # 计算每个类别的precision、recall、f1-score、support
        results_dict = {}
        results = ''
        for label in unique_labels:
            current_label_result = []
            # TP + FN
            tp_fn = y_trues.count(label)
            # TP + FP
            tp_fp = 0
            for y_pred in y_preds:
                if label in y_pred:
                    tp_fp += 1
            # TP
            tp = 0
            for i in range(len(y_trues)):
                if y_trues[i] == label and label in y_preds[i]:
                    tp += 1

            support = tp_fn

            try:
                precision = round(tp / tp_fp, digs)
                recall = round(tp / tp_fn, digs)
                f1_score = round(2 * (precision * recall) / (precision + recall), digs)
            except ZeroDivisionError:
                precision = 0
                recall = 0
                f1_score = 0

            current_label_result.append(precision)
            current_label_result.append(recall)
            current_label_result.append(f1_score)
            current_label_result.append(support)
            # 输出第一行
            results_dict[str(label)] = current_label_result
        title = '\t' + 'precision@' + str(k) + '\t' + 'recall@' + str(k) + '\t' + 'f1_score@' + str(
            k) + '\t' + 'support' + '\n'
        results += title

        for k, v in sorted(results_dict.items()):
            current_line = str(k) + '\t\t' + str(v[0]) + '\t\t' + str(v[1]) + '\t\t' + str(v[2]) + '\t\t' + str(
                v[3]) + '\n'
            results += current_line
        sums = len(y_trues)

        # 注意macro avg和weighted avg计算方式的不同
        macro_avg_results = [(v[0], v[1], v[2]) for k, v in sorted(results_dict.items())]
        weighted_avg_results = [(v[0] * v[3], v[1] * v[3], v[2] * v[3]) for k, v in sorted(results_dict.items())]

        # 计算macro avg
        macro_precision = 0
        macro_recall = 0
        macro_f1_score = 0
        for macro_avg_result in macro_avg_results:
            macro_precision += macro_avg_result[0]
            macro_recall += macro_avg_result[1]
            macro_f1_score += macro_avg_result[2]
        macro_precision /= num_classes
        macro_recall /= num_classes
        macro_f1_score /= num_classes

        # 计算weighted avg
        weighted_precision = 0
        weighted_recall = 0
        weighted_f1_score = 0
        for weighted_avg_result in weighted_avg_results:
            weighted_precision += weighted_avg_result[0]
            weighted_recall += weighted_avg_result[1]
            weighted_f1_score += weighted_avg_result[2]

        weighted_precision /= sums
        weighted_recall /= sums
        weighted_f1_score /= sums

        macro_avg_line = 'macro avg' + '\t\t' + str(round(macro_precision, digs)) + '\t\t' + str(
            round(macro_recall, digs)) + '\t\t' + str(round(macro_f1_score, digs)) + '\t\t' + str(sums) + '\n'
        weighted_avg_line = 'weighted avg' + '\t\t' + str(round(weighted_precision, digs)) + '\t\t' + str(
            round(weighted_recall, digs)) + '\t\t' + str(round(weighted_f1_score, digs)) + '\t\t' + str(sums)
        results += macro_avg_line
        results += weighted_avg_line
        return results, weighted_precision, weighted_recall, weighted_f1_score

    def run(self):
        test_data_set = rped.RPEDataset(self.data)
        test_data_loader = DataLoader(dataset=test_data_set, batch_size=1, shuffle=True)
        if self.mode == 1:
            # 用户轨迹列表
            user_list = []
            for i, user in enumerate(test_data_loader):
                user_list.append(user)
            # 提取所有用户轨迹的位置信息
            tra_true = []
            tra_pred = []
            for user in user_list:
                trace_ids = user.keys()
                for trace_id in trace_ids:
                    trace = user[trace_id]
                    tmp = []
                    for j in range(len(trace['tra_true'])):
                        tmp.append(trace['tra_true'][j].item())
                    tra_true.append(tmp)
                    tmp = []
                    for j in range(len(trace['tra_pred'])):
                        tmp.append(trace['tra_pred'][j].item())
                    tra_pred.append(tmp)
            # print(tra_true)
            # print(tra_pred)

            data_len = len(tra_pred)
            
            sumPrecison = 0
            sumRecall = 0
            sumF1 = 0
            for i in range(data_len):
                precision, recall, F1 = self.calMetric(tra_true[i], tra_pred[i])
                sumPrecison = sumPrecison + precision
                sumRecall = sumRecall + recall
                sumF1 = sumF1 + F1
            print('Precison', end=' ')
            print('%.2f' % (sumPrecison / data_len))
            print('Recall', end=' ')
            print('%.2f' % (sumRecall / data_len))
            print('F1', end=' ')
            print('%.2f' % (sumF1 / data_len))

            sumEdit = 0
            for i in range(data_len):
                edit = self.editDist(tra_true[i], tra_pred[i])
                sumEdit = sumEdit + edit
            print('Edit Distance', end=' ')
            print(sumEdit)
        else:
            # 用户轨迹列表
            user_list = []
            for i, user in enumerate(test_data_loader):
                user_list.append(user)
            # 提取所有用户轨迹的位置信息
            tra_true = []
            tra_pred = []
            for user in user_list:
                trace_ids = user.keys()
                for trace_id in trace_ids:
                    trace = user[trace_id]
                    tmp = []
                    for j in range(len(trace['tra_true'])):
                        tmp.append(trace['tra_true'][j].item())
                    tra_true.append(tmp)
                    tmp = []
                    for j in range(len(trace['tra_pred'])):
                        tmpIn = []
                        for i in range(len(trace['tra_pred'][j])):
                            tmpIn.append(trace['tra_pred'][j][i].item())
                        tmp.append(tmpIn)
                    tra_pred.append(tmp)
            # print(tra_true)
            # print(tra_pred)

            data_len = len(tra_pred)

            sumPrecison = 0
            sumRecall = 0
            sumF1 = 0
            for i in range(data_len):
                res, precision, recall, F1 = self.precision_recall_fscore_k(tra_true[i], tra_pred[i], k=self.k)
                sumPrecison = sumPrecison + precision
                sumRecall = sumRecall + recall
                sumF1 = sumF1 + F1
                # print(res)
                # print('------------------------------------------------')
            print('Precison@{}'.format(self.k), end=' ')
            print('%.2f' % (sumPrecison / data_len))
            print('Recall@{}'.format(self.k), end=' ')
            print('%.2f' % (sumRecall / data_len))
            print('F1@{}'.format(self.k), end=' ')
            print('%.2f' % (sumF1 / data_len))
