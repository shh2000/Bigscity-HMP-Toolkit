import shutil

from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys

from evaluate import loc_pred_evaluate_data as lped

sys.path.append('..')


class LocationPredEvaluate(object):
    def __init__(self, data, datatype='DeepMove', mode='ACC', k=1, len_pred=1, max_epoch=1, save_data=True):
        # 传入的数据参数以及评估参数
        self.data = data
        self.datatype = datatype
        self.mode = mode
        self.k = k
        self.len_pred = len_pred
        self.max_epoch = max_epoch
        self.epoch = 0
        self.base_dir = '..\\evaluate\\'
        # 保存的评估结果
        self.save_data = save_data
        self.metrics = {'trace': [], 'acc': [], 'loss': []}

    def init_data(self):
        """
        Here we transform specific data types to standard input type
        """

        # 加载json类型的数据为字典类型
        if type(self.data) == str:
            self.data = json.loads(self.data)
        assert type(self.data) == dict, "Illegal data type."

        # 对对应模型的输出转换格式
        if self.datatype == 'DeepMove':
            user_idx = self.data.keys()
            for user_id in user_idx:
                trace_idx = self.data[user_id].keys()
                for trace_id in trace_idx:
                    # print(user_id, trace_id)
                    trace = self.data[user_id][trace_id]
                    loc_pred = trace['loc_pred']
                    new_loc_pred = []
                    for t_list in loc_pred:
                        new_loc_pred.append(self.sort_confidence_ids(t_list))
                    self.data[user_id][trace_id]['loc_pred'] = new_loc_pred

    def sort_confidence_ids(self, confidence_list):
        """
        Here we convert the prediction results of the DeepMove model
        DeepMove model output: confidence of all locations
        Evaluate model input: location ids based on confidence
        :param confidence_list:
        :return: ids_list
        """
        assert self.datatype == 'DeepMove', 'call function sort_confidence_ids.'
        sorted_list = sorted(confidence_list, reverse=True)
        mark_list = [0 for i in confidence_list]
        ids_list = []
        for item in sorted_list:
            for i in range(len(confidence_list)):
                if confidence_list[i] == item and mark_list[i] == 0:
                    mark_list[i] = 1
                    ids_list.append(i)
                    break
            if len(ids_list) == self.k:
                self.len_pred = self.k
                break
        return ids_list

    # 均方误差（Mean Square Error）
    @staticmethod
    def MSE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'MSE: 预测数据与真实数据大小不一致'
        return np.mean(sum(pow(loc_pred - loc_true, 2)))

    # 平均绝对误差（Mean Absolute Error）
    @staticmethod
    def MAE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'MAE: 预测数据与真实数据大小不一致'
        return np.mean(sum(loc_pred - loc_true))

    # 均方根误差（Root Mean Square Error）
    @staticmethod
    def RMSE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'RMSE: 预测数据与真实数据大小不一致'
        return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))

    # 平均绝对百分比误差（Mean Absolute Percentage Error）
    @staticmethod
    def MAPE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'MAPE: 预测数据与真实数据大小不一致'
        assert 0 not in loc_true, "MAPE: 真实数据有0，该公式不适用"
        return np.mean(abs(loc_pred - loc_true) / loc_true)

    # 平均绝对和相对误差（Mean Absolute Relative Error）
    @staticmethod
    def MARE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), "MARE：预测数据与真实数据大小不一致"
        assert np.sum(loc_true) != 0, "MARE：真实位置全为0，该公式不适用"
        return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)

    # 对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）
    @staticmethod
    def SMAPE(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), 'SMAPE: 预测数据与真实数据大小不一致'
        assert 0 in (loc_pred + loc_true), "SMAPE: 预测数据与真实数据有0，该公式不适用"
        return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) + np.abs(loc_true)))

    # 对比真实位置与预测位置获得预测准确率
    @staticmethod
    def ACC(loc_pred, loc_true):
        assert len(loc_pred) == len(loc_true), "accuracy: 预测数据与真实数据大小不一致"
        loc_diff = loc_pred - loc_true
        loc_diff[loc_diff != 0] = 1
        return loc_diff, np.mean(loc_diff == 0)

    def topk(self, loc_pred, loc_true):
        assert self.k > 0, "top-k ACC评估方法：k值应不小于1"
        assert len(loc_pred) >= self.k, "top-k ACC评估方法：没有提供足够的预测数据做评估"
        assert len(loc_pred[0]) == len(loc_true), "top-k ACC评估方法：预测数据与真实数据大小不一致"
        if self.k == 1:
            t, avg_acc = self.ACC(loc_pred[0], loc_true)
            return avg_acc
        else:
            tot_list = np.zeros(len(loc_true), dtype=int)
            for i in range(self.k):
                t, avg_acc = self.ACC(loc_pred[i], loc_true)
                tot_list = tot_list + t
            return np.mean(tot_list < self.k)

    def run_evaluate(self, loc_pred, loc_true, field):
        assert len(loc_pred) == len(loc_true), "位置预测评估: 预测数据与真实数据大小不一致"
        loc_pred2 = [[] for j in range(self.len_pred)]
        for i in range(len(loc_true)):
            for j in range(self.len_pred):
                loc_pred2[j].append(loc_pred[i][j])
        if self.mode == 'ACC':
            avg_acc = self.topk(np.array(loc_pred2, dtype=object), np.array(loc_true, dtype=object))
            if field == 'model':
                print('---- 该模型在 top-{} ACC 评估方法下 avg_acc={:.3f} ----'.format(self.k, avg_acc))
                self.metrics['acc'].append(avg_acc)
            else:
                print('accuracy={:.3f}'.format(avg_acc))
                self.metrics['trace'].append(avg_acc)
        elif self.mode == 'RMSE':
            avg_loss = self.RMSE(np.array(loc_pred2[0]), np.array(loc_true))
            if field == 'model':
                print('---- 该模型在 RMSE 评估方法下 avg_loss={:.3f} ----'.format(avg_loss))
                self.metrics['loss'].append(avg_loss)
            else:
                print('RMSE avg_loss={:.3f}'.format(avg_loss))
                self.metrics['trace'].append(avg_loss)
        elif self.mode == "MSE":
            avg_loss = self.MSE(np.array(loc_pred2[0]), np.array(loc_true))
            if field == 'model':
                print('---- 该模型在 MSE 评估方法下 avg_loss={:.3f} ----'.format(avg_loss))
                self.metrics['loss'].append(avg_loss)
            else:
                print('MSE avg_loss={:.3f}'.format(avg_loss))
                self.metrics['trace'].append(avg_loss)
        elif self.mode == "MAE":
            avg_loss = self.MAE(np.array(loc_pred2[0]), np.array(loc_true))
            if field == 'model':
                print('---- 该模型在 MAE 评估方法下 avg_loss={:.3f} ----'.format(avg_loss))
                self.metrics['loss'].append(avg_loss)
            else:
                print('MAE avg_loss={:.3f}'.format(avg_loss))
                self.metrics['trace'].append(avg_loss)
        elif self.mode == "MAPE":
            avg_loss = self.MAPE(np.array(loc_pred2[0]), np.array(loc_true))
            if field == 'model':
                print('---- 该模型在 MAPE 评估方法下 avg_loss={:.3f} ----'.format(avg_loss))
                self.metrics['loss'].append(avg_loss)
            else:
                print('MAPE avg_loss={:.3f}'.format(avg_loss))
                self.metrics['trace'].append(avg_loss)
        elif self.mode == "MARE":
            avg_loss = self.MARE(np.array(loc_pred2[0]), np.array(loc_true))
            if field == 'model':
                print('---- 该模型在 MARE 评估方法下 avg_loss={:.3f} ----'.format(avg_loss))
                self.metrics['loss'].append(avg_loss)
            else:
                print('MARE avg_loss={:.3f}'.format(avg_loss))
                self.metrics['trace'].append(avg_loss)

    def run(self, data=None):
        """
        评估入口，根据评估类的参数做相应的评估
        :return:
        """

        self.epoch = self.epoch + 1
        if data is not None:
            self.data = data
        self.init_data()
        if self.save_data:
            if self.epoch == 1 and os.path.exists(self.base_dir + 'temp_data\\' + self.datatype):
                shutil.rmtree(self.base_dir + 'temp_data\\' + self.datatype)
            if not os.path.exists(self.base_dir + 'temp_data\\' + self.datatype):
                os.makedirs(self.base_dir + 'temp_data\\' + self.datatype)
            file = open(self.base_dir + 'temp_data\\' + self.datatype + '\\data_' + str(self.epoch) + '.json', 'w')
            file.write(json.dumps(self.data))
            file.close()
        test_data_set = lped.LPEDataset(self.data)
        test_data_loader = DataLoader(dataset=test_data_set, batch_size=1, shuffle=True)
        # 用户轨迹列表
        user_list = {}
        for i, user in enumerate(test_data_loader):
            for user_id in user.keys():
                user_list[user_id] = user[user_id]
        # 提取所有用户轨迹的位置信息
        loc_true = []
        loc_pred = []
        for user_id in user_list.keys():
            user = user_list[user_id]
            print('-------------- user_id {} --------------'.format(user_id))
            trace_ids = user.keys()
            for trace_id in trace_ids:
                trace = user[trace_id]
                t_loc_true = trace['loc_true']
                t_loc_pred = trace['loc_pred']
                loc_true.extend(t_loc_true)
                loc_pred.extend(t_loc_pred)
                print('trace_id {} : '.format(trace_id), end="")
                self.run_evaluate(t_loc_pred, t_loc_true, 'trace')
        self.run_evaluate(loc_pred, loc_true, 'model')
        if self.epoch == self.max_epoch:
            if not os.path.exists(self.base_dir + 'res\\' + self.datatype):
                os.makedirs(self.base_dir + 'res\\' + self.datatype)
            file = open(self.base_dir + 'res\\' + self.datatype + '\\res.json', 'w')
            file.write(json.dumps(self.metrics))
            file.close()

    def get_acc(self):
        assert self.epoch > 0, "还未对数据进行评估，请运行模型的run方法"
        assert self.epoch == self.max_epoch, "数据还没有评估完，请继续执行run方法"
        assert self.mode == "ACC", self.mode + "评估方法对应的指标为avg_loss"
        return self.metrics['acc']

    def get_loss(self):
        assert self.epoch > 0, "还未对数据进行评估，请运行模型的run方法"
        assert self.epoch == self.max_epoch, "数据还没有评估完，请继续执行run方法"
        assert self.mode != "ACC", self.mode + "评估方法对应的指标为avg_acc"
        return self.metrics['loss']

    def get_acc_list(self):
        assert self.epoch > 0, "还未对数据进行评估，请运行模型的run方法"
        assert self.epoch == self.max_epoch, "数据还没有评估完，请继续执行run方法"
        assert self.mode == "ACC", self.mode + "评估方法对应的指标为avg_loss"
        return self.metrics['trace']

    def get_loss_list(self):
        assert self.epoch > 0, "还未对数据进行评估，请运行模型的run方法"
        assert self.epoch == self.max_epoch, "数据还没有评估完，请继续执行run方法"
        assert self.mode != "ACC", self.mode + "评估方法对应的指标为avg_acc"
        return self.metrics['trace']
