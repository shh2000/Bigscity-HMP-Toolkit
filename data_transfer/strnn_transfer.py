import datetime
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import json


def load_data(train, read_line=6500000):
    user2id = {}
    poi2id = {}

    train_user = []
    train_time = []
    train_lati = []
    train_longi = []
    train_loc = []
    valid_user = []
    valid_time = []
    valid_lati = []
    valid_longi = []
    valid_loc = []
    test_user = []
    test_time = []
    test_lati = []
    test_longi = []
    test_loc = []

    train_f = open(train, 'r')
    lines = train_f.readlines()[:read_line]

    user_time = []
    user_lati = []
    user_longi = []
    user_loc = []
    visit_thr = 30

    prev_user = int(lines[0].split('\t')[0])
    visit_cnt = 0
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user == prev_user:
            visit_cnt += 1
        else:
            if visit_cnt >= visit_thr:  # 只记录超过30行数据的user
                # 记录每个user的序号 因为会忽略数据少于30的user 所以序号变了
                user2id[prev_user] = len(user2id)
            prev_user = user
            visit_cnt = 1

    train_f = open(train, 'r')
    lines = train_f.readlines()[:read_line]

    prev_user = int(lines[0].split('\t')[0])
    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        if user2id.get(user) is None:
            continue
        user = user2id.get(user)

        time = (datetime.datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")
                - datetime.datetime(2009, 1, 1)).total_seconds() / 60  # minutes
        lati = float(tokens[2])
        longi = float(tokens[3])
        location = int(tokens[4])
        if poi2id.get(location) is None:
            poi2id[location] = len(poi2id)  # 记录每个位置的序号
        location = poi2id.get(location)  # 把数据集中原始的位置修改成自定的编号

        if user == prev_user:
            user_time.insert(0, time)
            user_lati.insert(0, lati)
            user_longi.insert(0, longi)
            user_loc.insert(0, location)
        else:
            train_thr = int(len(user_time) * 0.7)
            valid_thr = int(len(user_time) * 0.8)
            train_user.append(user)  # 源代码是user 我觉得是prev_user
            train_time.append(user_time[:train_thr])
            train_lati.append(user_lati[:train_thr])
            train_longi.append(user_longi[:train_thr])
            train_loc.append(user_loc[:train_thr])
            valid_user.append(user)
            valid_time.append(user_time[train_thr:valid_thr])
            valid_lati.append(user_lati[train_thr:valid_thr])
            valid_longi.append(user_longi[train_thr:valid_thr])
            valid_loc.append(user_loc[train_thr:valid_thr])
            test_user.append(user)
            test_time.append(user_time[valid_thr:])
            test_lati.append(user_lati[valid_thr:])
            test_longi.append(user_longi[valid_thr:])
            test_loc.append(user_loc[valid_thr:])

            prev_user = user
            user_time = [time]
            user_lati = [lati]
            user_longi = [longi]
            user_loc = [location]

    if user2id.get(user) is not None:
        train_thr = int(len(user_time) * 0.7)
        valid_thr = int(len(user_time) * 0.8)
        train_user.append(user)
        train_time.append(user_time[:train_thr])
        train_lati.append(user_lati[:train_thr])
        train_longi.append(user_longi[:train_thr])
        train_loc.append(user_loc[:train_thr])
        valid_user.append(user)
        valid_time.append(user_time[train_thr:valid_thr])
        valid_lati.append(user_lati[train_thr:valid_thr])
        valid_longi.append(user_longi[train_thr:valid_thr])
        valid_loc.append(user_loc[train_thr:valid_thr])
        test_user.append(user)
        test_time.append(user_time[valid_thr:])
        test_lati.append(user_lati[valid_thr:])
        test_loc.append(user_loc[valid_thr:])

    f = open('.strnn/train_file.csv', 'w')
    f.write('useid' + '\t' + 'time' + '\t' + 'lat' + '\t' + 'lon' + '\t' + 'locid' + '\n')
    for i in range(len(train_user)):
        for j in range(len(train_time[i])):
            f.write(str(train_user[i]) + '\t' + str(train_time[i][j]) + '\t'
                    + str(train_lati[i][j]) + '\t' + str(train_longi[i][j]) + '\t' + str(train_loc[i][j]) + '\n')
    f.close()

    f = open('.strnn/test_file.csv', 'w')
    f.write('useid' + '\t' + 'time' + '\t' + 'lat' + '\t' + 'lon' + '\t' + 'locid' + '\n')
    for i in range(len(test_user)):
        for j in range(len(test_time[i])):
            f.write(str(test_user[i]) + '\t' + str(test_time[i][j]) + '\t'
                    + str(test_lati[i][j]) + '\t' + str(test_longi[i][j]) + '\t' + str(test_loc[i][j]) + '\n')
    f.close()

    f = open('.strnn/valid_file.csv', 'w')
    f.write('useid' + '\t' + 'time' + '\t' + 'lat' + '\t' + 'lon' + '\t' + 'locid' + '\n')
    for i in range(len(valid_user)):
        for j in range(len(valid_time[i])):
            f.write(str(valid_user[i]) + '\t' + str(valid_time[i][j]) + '\t'
                    + str(valid_lati[i][j]) + '\t' + str(valid_longi[i][j]) + '\t' + str(valid_loc[i][j]) + '\n')
    f.close()

    return len(user2id), poi2id, train_user, train_time, train_lati,\
           train_longi, train_loc, valid_user, valid_time, valid_lati, \
           valid_longi, valid_loc, test_user, test_time, test_lati, test_longi, test_loc


# def inner_iter(data, batch_size):
#     data_size = len(data)
#     num_batches = int(len(data) / batch_size)
#     for batch_num in range(num_batches):
#         start_index = batch_num * batch_size
#         end_index = min((batch_num + 1) * batch_size, data_size)
#         yield data[start_index:end_index]


# Parameters
# ==================================================
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # winodw width (6h)
up_time = 1440  # 1d
lw_time = 50    # 50m
up_dist = 100   # ??
lw_dist = 1


class STRNNModule(nn.Module):
    def __init__(self):
        super(STRNNModule, self).__init__()
        # embedding:
        self.user_weight = Variable(torch.randn(user_cnt, dim), requires_grad=False).type(ftype)
        self.h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(ftype)
        self.location_weight = nn.Embedding(len(poi2id), dim)
        self.perm_weight = nn.Embedding(user_cnt, dim)
        # attributes:
        self.time_upper = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.time_lower = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.dist_upper = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.dist_lower = nn.Parameter(torch.randn(dim, dim).type(ftype))
        self.C = nn.Parameter(torch.randn(dim, dim).type(ftype))
        # modules:
        self.sigmoid = nn.Sigmoid()

    # find the most closest value to w, w_cap(index)
    def find_w_cap(self, times, i):
        trg_t = times[i] - ww
        tmp_t = times[i]
        tmp_i = i - 1
        for idx, t_w in enumerate(reversed(times[:i]), start=1):
            if t_w.data.cpu().numpy() == trg_t.data.cpu().numpy():
                return i - idx
            elif t_w.data.cpu().numpy() > trg_t.data.cpu().numpy():
                tmp_t = t_w
                tmp_i = i - idx
            elif t_w.data.cpu().numpy() < trg_t.data.cpu().numpy():
                if trg_t.data.cpu().numpy() - t_w.data.cpu().numpy() \
                        < tmp_t.data.cpu().numpy() - trg_t.data.cpu().numpy():
                    return i - idx
                else:
                    return tmp_i
        return 0

    def return_h_tw(self, times, latis, longis, locs, idx):
        w_cap = self.find_w_cap(times, idx)
        if w_cap is 0:
            return self.h_0
        else:
            self.return_h_tw(times, latis, longis, locs, w_cap)

        lati = latis[idx] - latis[w_cap:idx]
        longi = longis[idx] - longis[w_cap:idx]
        td = times[idx] - times[w_cap:idx]
        ld = self.euclidean_dist(lati, longi)

        data = ','.join(str(e) for e in td.data.cpu().numpy()) + "\t"
        f.write(data)
        data = ','.join(str(e) for e in ld.data.cpu().numpy()) + "\t"
        f.write(data)
        # data = ','.join(str(e.data.cpu().numpy()[0]) for e in locs[w_cap:idx])+"\t"
        data = ','.join(str(e.data.cpu().numpy()) for e in locs[w_cap:idx]) + "\t"
        f.write(data)
        # data = str(locs[idx].data.cpu().numpy()[0])+"\n"
        data = str(locs[idx].data.cpu().numpy()) + "\n"
        f.write(data)

    # get transition matrices by linear interpolation
    def get_location_vector(self, td, ld, locs):
        tud = up_time - td
        tdd = td - lw_time
        lud = up_dist - ld
        ldd = ld - lw_dist
        loc_vec = 0
        for i in range(len(tud)):
            Tt = torch.div(torch.mul(self.time_upper, tud[i])
                           + torch.mul(self.time_lower, tdd[i]), tud[i] + tdd[i])
            Sl = torch.div(torch.mul(self.dist_upper, lud[i])
                           + torch.mul(self.dist_lower, ldd[i]), lud[i] + ldd[i])
            loc_vec += torch.mm(Sl, torch.mm(Tt,
                                             torch.t(self.location_weight(locs[i]))))
        return loc_vec

    def euclidean_dist(self, x, y):
        return torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

    # neg_lati, neg_longi, neg_loc, step):
    # user, times, latis, longis, locs全是一维的
    def forward(self, user, times, latis, longis, locs, step):
        f.write(str(user.data.cpu().numpy()[0]) + "\n")
        # positive sampling
        pos_h = self.return_h_tw(times, latis, longis, locs, len(times) - 1)


def run(user, time, lati, longi, loc, step):
    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)
    time = Variable(torch.from_numpy(np.asarray(time))).type(ftype)
    lati = Variable(torch.from_numpy(np.asarray(lati))).type(ftype)
    longi = Variable(torch.from_numpy(np.asarray(longi))).type(ftype)
    loc = Variable(torch.from_numpy(np.asarray(loc))).type(ltype)
    rnn_output = strnn_model(user, time, lati, longi, loc, step)


# Data loading params
train_file = "../data_extract/datasets/Gowalla_totalCheckins.txt"

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
user_cnt, poi2id, train_user, train_time, train_lati, \
    train_longi, train_loc, valid_user, valid_time, valid_lati, \
    valid_longi, valid_loc, test_user, test_time, test_lati, \
    test_longi, test_loc = load_data(train_file, 5000)

print("User/Location: {:d}/{:d}".format(user_cnt, len(poi2id)))
di = {'user_cnt': user_cnt, 'loc_cnt': len(poi2id)}
json.dump(di, open('./strnn/config.json', 'w'))
print("==================================================================================")

strnn_model = STRNNModule().cuda()

print("Making train file...")
f = open("./strnn/prepro_train_%s.txt" % lw_time, 'w')
# Training
# 不同user的time,lat,lon,loc是不一样多的 这里进行了合并
# 效果是 time[0],lat[0],lon[0],loc[0]合并到一起  time[1],lat[1],lon[1],loc[1]合并到一起...
# 也就是每一个用户的相关数据合并到一个元组中 len(train_batches)=len(user)
train_batches = list(zip(train_time, train_lati, train_longi, train_loc))
# 这里进行分割 每个用户的数据分别训练
for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
    batch_time, batch_lati, batch_longi, batch_loc = train_batch  # 获取每个用户的四维数据
    run(train_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=1)  # 处理原始数据
f.close()

print("Making valid file...")
f = open("./strnn/prepro_valid_%s.txt" % lw_time, 'w')
# Eavludating
valid_batches = list(zip(valid_time, valid_lati, valid_longi, valid_loc))
for j, valid_batch in enumerate(tqdm.tqdm(valid_batches, desc="valid")):
    batch_time, batch_lati, batch_longi, batch_loc = valid_batch  # inner_batch)
    run(valid_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=2)
f.close()

print("Making test file...")
f = open("./strnn/prepro_test_%s.txt" % lw_time, 'w')
# Testing
test_batches = list(zip(test_time, test_lati, test_longi, test_loc))
for j, test_batch in enumerate(tqdm.tqdm(test_batches, desc="test")):
    batch_time, batch_lati, batch_longi, batch_loc = test_batch  # inner_batch)
    run(test_user[j], batch_time, batch_lati, batch_longi, batch_loc, step=3)
f.close()
