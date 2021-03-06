import sys,os
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import json
# import pickle
# 将父目录加入 sys path TODO: 有没有更好的引用方式
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('..')
from model import TrajPreLocalAttnLong, SimpleRNN
from data_transfer import gen_data
from utils import RnnParameterData, generate_history, transferModelToMode
from evaluate import loc_pred_evaluate_model as lpem
from datasets import DeepMoveDataset
from torch.utils.data import DataLoader

# get model name and datasets from command line 
if len(sys.argv) != 3:
    print('wrong format parameters!', file=sys.stderr)
    exit(1)
model_name = sys.argv[1] # deepMove / SimpleRNN / FPMC
datasets = sys.argv[2]
model_mode = transferModelToMode(model_name)

# get transfer parameters
f = open('../config/deepmove_args.json', 'r')
config = json.load(f)
f.close()
time_length = config['transfer']['time_length']
data = gen_data(model_name, datasets, config['transfer']['min_session_len'], config['transfer']['min_sessions'], time_length)
print('data loaded')
data_neural = data['data_neural']
use_cuda = config['train']['use_cuda']
# pad parameter
loc_pad = data['loc_size']
data['loc_size'] += 1
tim_pad = time_length
time_length += 1
pad_len = config['transfer']['pad_len']
history_len =  config['transfer']['history_len']
target_len = pad_len - history_len
test_dataset = DeepMoveDataset(data_neural, 'test', use_cuda, loc_pad, tim_pad, pad_len, history_len)

parameters = RnnParameterData(data=data, time_size=time_length, model_mode=model_mode, use_cuda = use_cuda)
SAVE_PATH = '../model/save_model/'
if model_name == 'deepMove':
    model = TrajPreLocalAttnLong(parameters=parameters).cuda() if use_cuda else TrajPreLocalAttnLong(parameters=parameters)
elif model_name == 'simpleRNN':
    model = SimpleRNN(parameters=parameters).cuda() if use_cuda else SimpleRNN(parameters=parameters)
if os.path.exists(SAVE_PATH + model_name + '.m'):
    model.load_state_dict(torch.load(SAVE_PATH + model_name + '.m'))
    print('load model')
else:
    print('no pretrained model! please train the model.')
model.train(False)

print('start evaluate')
batch_size = 100
num_workers = 0
total_batch = test_dataset.__len__() / batch_size
verbose = 10

# custom loader
def collactor(batch):
    loc = []
    tim = []
    history_loc = []
    history_tim = []
    history_count = []
    uid = []
    target = []
    session_id = []
    for item in batch:
        loc.append(item['loc'])
        tim.append(item['tim'])
        history_loc.append(item['history_loc'])
        history_tim.append(item['history_tim'])
        history_count.append(item['history_count'])
        uid.append(item['uid'])
        target.append(item['target'])
        session_id.append(item['session_id'])
    return loc, tim, history_loc, history_tim, history_count, uid, target, session_id

test_data_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, num_workers = num_workers, collate_fn = collactor)
cnt = 0
total_batch = math.ceil(test_dataset.__len__() / batch_size) 
lpt = lpem.LocationPredEvaluate({}, 'DeepMove', 'ACC', 2, data['loc_size'], total_batch + 1) # TODO: 这里可以改一改
for loc, tim, history_loc, history_tim, history_count, uid, target, session_id in test_data_loader:
    if use_cuda:
        loc = torch.LongTensor(loc).cuda()
        tim = torch.LongTensor(tim).cuda()
        target = torch.LongTensor(target).cuda()
    else:
        loc = torch.LongTensor(loc)
        tim = torch.LongTensor(tim)
        target = torch.LongTensor(target)
    if model_mode == 'attn_local_long':
        scores = model(loc, tim, target_len)
    elif model_mode == 'simple':
        scores = model(loc, tim)
        scores = scores[:, -target_len:, :]
    evaluate_input = {}
    for i in range(len(uid)):
        u = uid[i]
        s = session_id[i]
        trace_input = {}
        trace_input['loc_true'] = target[i].tolist()
        trace_input['loc_pred'] = scores[i].tolist()
        if u not in evaluate_input:
            evaluate_input[u] = {}
        evaluate_input[u][s] = trace_input
        lpt.run(evaluate_input)
    cnt += 1
    if cnt % verbose == 0:
        print('finish batch {}/{}'.format(cnt, total_batch))
