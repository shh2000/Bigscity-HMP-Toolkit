import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

# 将父目录加入 sys path TODO: 有没有更好的引用方式
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('..')
from model import TrajPreLocalAttnLong
from data_transfer import gen_data
from utils import RnnParameterData, run, generate_history, transferModelToMode
from datasets import DeepMoveDataset

# get model name and datasets from command line 
if len(sys.argv) != 3:
    print('wrong format parameters!', file=sys.stderr)
    exit(1)
model_name = sys.argv[1] # deepMove / SimpleRNN
datasets = sys.argv[2]
model_mode = transferModelToMode(model_name)

# read config from ../config/deepmove_args.json
f = open('../config/deepmove_args.json', 'r')
config = json.load(f)
f.close()
time_length = config['transfer']['time_length']
data = gen_data(model_name, datasets, config['transfer']['min_session_len'], config['transfer']['min_sessions'], time_length)
print('data loaded')
data_neural = data['data_neural']

use_cuda = config['train']['use_cuda']
parameters = RnnParameterData(data=data, time_size=time_length, model_mode=model_mode, use_cuda = use_cuda)

if use_cuda:
    model = TrajPreLocalAttnLong(parameters=parameters).cuda()
    criterion = nn.NLLLoss().cuda()
else:
    model = TrajPreLocalAttnLong(parameters=parameters)
    criterion = nn.NLLLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                       weight_decay=parameters.L2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                 factor=parameters.lr_decay, threshold=1e-3)
lr = parameters.lr
# use dataset
train_dataset = DeepMoveDataset(data_neural, 'train', use_cuda)
test_dataset = DeepMoveDataset(data_neural, 'test', use_cuda)

SAVE_PATH = '../model/save_model/'
tmp_path = 'checkpoint/'
os.mkdir(SAVE_PATH + tmp_path)
print('start train')
## tran parameter
batch_size = 10
num_workers = 0
total_batch = train_dataset.__len__() / batch_size
metrics = {}
metrics['train_loss'] = []
# custom loader
def collactor(batch):
    loc = []
    tim = []
    history_loc = []
    history_tim = []
    history_count = []
    uid = []
    target = []
    target_len = []
    for item in batch:
        loc.append(item['loc'])
        tim.append(item['tim'])
        history_loc.append(item['history_loc'])
        history_tim.append(item['history_tim'])
        history_count.append(item['history_count'])
        uid.append(item['uid'])
        target.append(item['target'])
        target_len.append(item['target_len'])
    return loc, tim, history_loc, history_tim, history_count, uid, target_len, target

for epoch in range(parameters.epoch):
    ## train stage
    train_data_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, num_workers = num_workers, collate_fn = collactor)
    model, avg_loss = run(train_data_loader, model, optimizer, criterion, model_mode, lr, parameters.clip, parameters.use_cuda)
    print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
    metrics['train_loss'].append(avg_loss)
    ## test stage
    # TODO: find a test way
    # test_data_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, num_workers = num_workers, collate_fn = collactor)
    # print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))
    # metrics['valid_loss'].append(avg_loss)
    # metrics['accuracy'].append(avg_acc)
    # metrics['valid_acc'][epoch] = users_acc
    # save_name_tmp = 'ep_' + str(epoch) + '.m'
    # torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)
    # scheduler.step(avg_acc)
    # lr_last = lr
    # lr = optimizer.param_groups[0]['lr']
    # if lr_last > lr:
    #     load_epoch = np.argmax(metrics['accuracy'])
    #     load_name_tmp = 'ep_' + str(load_epoch) + '.m'
    #     model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
    #     print('load epoch={} model state'.format(load_epoch))
    if lr <= 0.9 * 1e-5:
        break

# mid = np.argmax(metrics['accuracy'])  # 这个不是最好的一次吗？
# avg_acc = metrics['accuracy'][mid]
# load_name_tmp = 'ep_' + str(mid) + '.m'
# model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
torch.save(model.state_dict(), SAVE_PATH + model_name + '.m')
# 删除之前创建的临时文件夹
for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
os.rmdir(SAVE_PATH + tmp_path)
