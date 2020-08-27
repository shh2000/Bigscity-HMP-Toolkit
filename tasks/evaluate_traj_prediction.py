import sys,os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
# import pickle
# 将父目录加入 sys path TODO: 有没有更好的引用方式
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('..')
from model import TrajPreLocalAttnLong, SimpleRNN
from data_transfer import gen_data
from utils import RnnParameterData, generate_history
from evaluate import loc_pred_evaluate_model as lpem

################################
# TODO: user define arg, should pass form command line
min_session_len = 5
min_sessions = 2
time_length = 72
datasets = 'traj_foursquare'
# TODO: 这部分仍旧照搬 deepMove 的源码，不太适配还需要进一步改进
model_name = 'simpleRNN' # now support deepMove/simpleRNN
model_mode = 'simple' # simple or attn_local_long
use_cuda = True
#################################
data = gen_data(model_name, datasets) # 不传上述三个参数时，将使用默认数值
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
print('data len: ', len(data_train))
# TODO:这里应该加载预训练好保存了的参数
# 但目前就先这样吧, for test
parameters = RnnParameterData(data=data, time_size=time_length, model_mode=model_mode, use_cuda = use_cuda)
if model_name == 'deepMove':
    model = TrajPreLocalAttnLong(parameters=parameters).cuda() if use_cuda else TrajPreLocalAttnLong(parameters=parameters)
else:
    model = SimpleRNN(parameters=parameters).cuda() if use_cuda else SimpleRNN(parameters=parameters)
model.train(False)

print('start evaluate')
evaluate_input = {}
verbose = 100
cnt = 1
for user in data_train.keys():
    evaluate_input[user] = {}
    if cnt % verbose == 0:
        print('start {} user: {}'.format(cnt, user))
    for session_id in data_train[user].keys():
        session = data_train[user][session_id]
        if model_name == 'deepMove':
            # 对于 TrajPreLocalAttnLong 是这个参数
            if use_cuda:
                loc = session['loc'].cuda()
                tim = session['tim'].cuda()
                target = session['target'].cuda()
            else:
                loc = session['loc']
                tim = session['tim']
                target = session['target']
            # uid = Variable(torch.LongTensor([user]))
            target_len = target.data.size()[0]
            scores = model(loc, tim, target_len)
            trace_input = {}
            trace_input['loc_true'] = target.tolist()
            trace_input['loc_pred'] = scores.tolist()
            evaluate_input[user][session_id] = trace_input
        elif model_name == 'simpleRNN':
            if use_cuda:
                loc = session['loc'].cuda()
                tim = session['tim'].cuda()
                target = session['target'].cuda()
            else:
                loc = session['loc']
                tim = session['tim']
                target = session['target']
            scores = model(loc, tim)
            scores = scores[-target.data.size()[0]:]
            trace_input = {}
            trace_input['loc_true'] = target.tolist()
            trace_input['loc_pred'] = scores.tolist()
            evaluate_input[user][session_id] = trace_input
    cnt += 1
    

lpt = lpem.LocationPredEvaluate(evaluate_input, "ACC", 2, data['loc_size'])
lpt.run()
