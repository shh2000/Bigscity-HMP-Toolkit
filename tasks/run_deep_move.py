import sys,os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
# 将父目录加入 sys path TODO: 有没有更好的引用方式
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TrajPreLocalAttnLong
from data_transfer import gen_data
from utils import RnnParameterData, run_simple

def generate_history(data_neural, mode):
    # use this to gen train data and test data
    data_train = {}
    train_idx = {}
    user_set = data_neural.keys()
    for u in user_set:
        if mode == 'test' and len(data_neural[u][mode]) == 0:
            # 当一用户 session 过少时会发生这个现象
            continue
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            target = np.array([s[0] for s in session[1:]])
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history_tim = [t[1] for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1
            history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1)) # 把多个 history 路径合并成一个？
            history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
            trace['history_loc'] = Variable(torch.LongTensor(history_loc))
            trace['history_tim'] = Variable(torch.LongTensor(history_tim))
            trace['history_count'] = history_count
            loc_tim = history
            loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
            loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
            tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
            trace['loc'] = Variable(torch.LongTensor(loc_np)) # loc 会与 history loc 有重合， loc 的前半部分为 history loc
            trace['tim'] = Variable(torch.LongTensor(tim_np))
            trace['target'] = Variable(torch.LongTensor(target)) # target 会与 loc 有一段的重合，只有 target 的最后一位 loc 没有
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

data = gen_data('deepMove', 'small_sample')
parameters = RnnParameterData(data=data)
model = TrajPreLocalAttnLong(parameters=parameters).cuda()
criterion = nn.NLLLoss().cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=parameters.lr,
                           weight_decay=parameters.L2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=parameters.lr_step,
                                                    factor=parameters.lr_decay, threshold=1e-3)
lr = parameters.lr

data_train, train_idx = generate_history(parameters.data_neural, 'train')
data_test, test_idx = generate_history(parameters.data_neural, 'test')

SAVE_PATH = './save_model/'
tmp_path = 'checkpoint/'
os.mkdir(SAVE_PATH + tmp_path)

for epoch in range(parameters.epoch):
    start_time = time.time()     
    model, avg_loss = run_simple(data_train, train_idx, 'train', lr, parameters.clip, model, optimizer,
                                         criterion, parameters.model_mode)
    print('==>Train Epoch:{:0>2d} Loss:{:.4f} lr:{}'.format(epoch, avg_loss, lr))
    metrics['train_loss'].append(avg_loss)
    avg_loss, avg_acc, users_acc = run_simple(data_test, test_idx, 'test', lr, parameters.clip, model,
                                                  optimizer, criterion, parameters.model_mode)
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

mid = np.argmax(metrics['accuracy']) # 这个不是最好的一次吗？
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