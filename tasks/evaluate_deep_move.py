import sys,os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
# 将父目录加入 sys path TODO: 有没有更好的引用方式
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('..')
from model import TrajPreLocalAttnLong
from data_transfer import gen_data
from utils import RnnParameterData, generate_history
from evaluate import loc_pred_evaluate_model as lpem

data = gen_data('deepMove', 'traj_foursquare')
print('data load')
data_neural = data['data_neural']
# 使用 deepMove 源码数据集进行调试
# picklefile=open('../data_extract/datasets/foursquare.pk','rb')
# data=pickle.load(picklefile,encoding='iso-8859-1')
# picklefile.close()
# data_neural = data['data_neural']
# data['loc_size'] = len(data['vid_list'])
# data['uid_size'] = len(data['uid_list'])
# TODO: generate_history 应该还需要修改一下
data_train, train_idx = generate_history(data_neural, 'train') # TODO: 评估的话不需要分 train 与 test
# TODO:这里应该加载预训练好保存了的参数
# 但目前就先这样吧, for test
parameters = RnnParameterData(data=data)
model = TrajPreLocalAttnLong(parameters=parameters).cuda()
model.train(False)

print('start evaluate')
evaluate_input = {}
verbose = 100
cnt = 1
for user in data_train.keys():
    evaluate_input[user] = {}
    if cnt % verbose == 0:
        print('start {} user: {}'.format(cnt, user))
    if cnt == 200:
        # TODO: solve memory error on big datasets
        break
    for session_id in data_train[user].keys():
        session = data_train[user][session_id]
        # 对于 TrajPreLocalAttnLong 是这个参数
        loc = session['loc'].cuda()
        tim = session['tim'].cuda()
        target = session['target'].cuda()
        # uid = Variable(torch.LongTensor([user]))
        target_len = target.data.size()[0]
        scores = model(loc, tim, target_len)
        trace_input = {}
        trace_input['loc_true'] = target.tolist()
        trace_input['loc_pred'] = scores.tolist()
        evaluate_input[user][session_id] = trace_input
    
    cnt += 1
    

lpt = lpem.LocationPredEvaluate(evaluate_input, "DeepMove", "ACC", 2, data['loc_size'])
lpt.run()
