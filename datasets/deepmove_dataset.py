from torch.utils.data import Dataset
import torch
from torch.autograd import Variable
import numpy as np

class DeepMoveDataset(Dataset):
    def __init__(self, data, mode, use_cuda):
        self.data = self.parseData(data, mode)
        self.use_cuda = use_cuda

    def __getitem__(self, index):
        item = self.data[index]
        if self.use_cuda:
            item['loc'] = item['loc'].cuda()
            item['tim'] = item['tim'].cuda()
            item['history_loc'] = item['history_loc'].cuda()
            item['history_tim'] = item['history_tim'].cuda()
            item['uid'] = item['uid'].cuda()
            item['target'] = item['target'].cuda()
        return item
        # return item['loc'], item['tim'], item['history_loc'], item['history_tim'], item['history_count'], item['uid'], item['target']

    def __len__(self):
        return len(self.data)
    
    def parseData(self, data_neural, mode):
        '''
        return list of data
        (loc, tim, history_loc, hisory_tim, history_count, uid, target)
        '''
        data = []
        user_set = data_neural.keys()
        for u in user_set:
            if mode == 'test' and len(data_neural[u][mode]) == 0:
                # 当一用户 session 过少时会发生这个现象
                continue
            sessions = data_neural[u]['sessions']
            if mode == 'all':
                train_id = data_neural[u]['train'] + data_neural[u]['test']
            else:
                train_id = data_neural[u][mode]
            for c, i in enumerate(train_id):
                trace = {}
                if mode == 'train' and c == 0 or mode == 'all' and c == 0:
                    continue
                session = sessions[i]
                if len(session) <= 1:
                    continue
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
                history_loc = np.reshape(np.array([s[0] for s in history]), (len(history), 1))  # 把多个 history 路径合并成一个？
                history_tim = np.reshape(np.array([s[1] for s in history]), (len(history), 1))
                trace['history_loc'] = Variable(torch.LongTensor(history_loc))
                trace['history_tim'] = Variable(torch.LongTensor(history_tim))
                trace['history_count'] = history_count
                loc_tim = history
                loc_tim.extend([(s[0], s[1]) for s in session[:-1]])
                loc_np = np.reshape(np.array([s[0] for s in loc_tim]), (len(loc_tim), 1))
                tim_np = np.reshape(np.array([s[1] for s in loc_tim]), (len(loc_tim), 1))
                trace['loc'] = Variable(torch.LongTensor(loc_np))  # loc 会与 history loc 有重合， loc 的前半部分为 history loc
                trace['tim'] = Variable(torch.LongTensor(tim_np))
                trace['target_len'] = len(target)
                trace['target'] = Variable(torch.LongTensor(target))  # target 会与 loc 有一段的重合，只有 target 的最后一位 loc 没有
                trace['uid'] = Variable(torch.LongTensor([int(u)]))
                trace['session_id'] = i
                data.append(trace)
        return data  