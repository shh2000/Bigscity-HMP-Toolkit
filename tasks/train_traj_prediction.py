import sys, os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

# 将父目录加入 sys path TODO: 有没有更好的引用方式
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('..')
from model import TrajPreLocalAttnLong
from data_transfer import gen_data
from utils import RnnParameterData, run_simple, generate_history, transferModelToMode

# get model name and datasets from command line 
if len(sys.argv) != 3:
    print('wrong format parameters!', file=sys.stderr)
    exit(1)
model_name = sys.argv[1] # deepMove / SimpleRNN
datasets = sys.argv[2]
model_mode = transferModelToMode(model_name)

# get transfer parameters
print('Please input transfer parameters. If you want to use default parameters, just enter.')

min_session_len = input('min session len:')
min_session_len = 5 if min_session_len == '' else min_session_len
min_sessions = input('min sessions:')
min_sessions = 2 if min_sessions == '' else min_sessions
time_length = input('time length:')
time_length = 72 if time_length == '' else time_length
if min_session_len == 5 and min_sessions == 2 and time_length == 72:
    data = gen_data(model_name, datasets)
else:    
    data = gen_data(model_name, datasets, min_session_len, min_sessions, time_length) # 不传上述三个参数时，将使用默认数值
print('data loaded')
data_neural = data['data_neural']

print('Please input model parameters. If you want to use default parameters, just enter.')
use_cuda = input('use cuda(yes/no):')
use_cuda = True if use_cuda == '' or use_cuda == 'yes' else False
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

data_train, train_idx = generate_history(parameters.data_neural, 'train')
data_test, test_idx = generate_history(parameters.data_neural, 'test')

SAVE_PATH = '../model/save_model/'
tmp_path = 'checkpoint/'
os.mkdir(SAVE_PATH + tmp_path)
print('start train')
for epoch in range(parameters.epoch):
    start_time = time.time()
    model, avg_loss = run_simple(data_train, train_idx, 'train', lr, parameters.clip, model, optimizer,
                                         criterion, parameters.model_mode, use_cuda)
    print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
    metrics['train_loss'].append(avg_loss)
    avg_loss, avg_acc, users_acc = run_simple(data_test, test_idx, 'test', lr, parameters.clip, model,
                                                  optimizer, criterion, parameters.model_mode, use_cuda)
    print('==>Test Acc:{:.4f} Loss:{:.4f}'.format(avg_acc, avg_loss))
    metrics['valid_loss'].append(avg_loss)
    metrics['accuracy'].append(avg_acc)
    metrics['valid_acc'][epoch] = users_acc
    save_name_tmp = 'ep_' + str(epoch) + '.m'
    torch.save(model.state_dict(), SAVE_PATH + tmp_path + save_name_tmp)
    scheduler.step(avg_acc)
    lr_last = lr
    lr = optimizer.param_groups[0]['lr']
    if lr_last > lr:
        load_epoch = np.argmax(metrics['accuracy'])
        load_name_tmp = 'ep_' + str(load_epoch) + '.m'
        model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
        print('load epoch={} model state'.format(load_epoch))
    if epoch == 0:
        print('single epoch time cost:{}'.format(time.time() - start_time))
    if lr <= 0.9 * 1e-5:
        break

mid = np.argmax(metrics['accuracy'])  # 这个不是最好的一次吗？
avg_acc = metrics['accuracy'][mid]
load_name_tmp = 'ep_' + str(mid) + '.m'
model.load_state_dict(torch.load(SAVE_PATH + tmp_path + load_name_tmp))
save_name = 'res'
json.dump({'args': argv, 'metrics': metrics}, fp=open(SAVE_PATH + save_name + '.rs', 'w'), indent=4)
torch.save(model.state_dict(), SAVE_PATH + save_name + '.m')
# 删除之前创建的临时文件夹
for rt, dirs, files in os.walk(SAVE_PATH + tmp_path):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
os.rmdir(SAVE_PATH + tmp_path)
