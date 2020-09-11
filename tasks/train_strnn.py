import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from model import STRNN
import json

def parameters():
    params = []
    for model in [strnn_model]:
        params += list(model.parameters())
    return params


def treat_prepro(train, step):
    train_f = open(train, 'r')
    # Need to change depending on threshold
    if step == 1:
        lines = train_f.readlines()  # [:86445] #659 #[:309931]
    elif step == 2:
        lines = train_f.readlines()  # [:13505]#[:309931]
    elif step == 3:
        lines = train_f.readlines()  # [:30622]#[:309931]

    train_user = []
    train_td = []
    train_ld = []
    train_loc = []
    train_dst = []

    user = 1
    user_td = []
    user_ld = []
    user_loc = []
    user_dst = []

    for i, line in enumerate(lines):
        tokens = line.strip().split('\t')
        if len(tokens) < 3:
            if user_td:
                train_user.append(user)
                train_td.append(user_td)
                train_ld.append(user_ld)
                train_loc.append(user_loc)
                train_dst.append(user_dst)
            user = int(tokens[0])
            user_td = []
            user_ld = []
            user_loc = []
            user_dst = []
            continue
        td = np.array([float(t) for t in tokens[0].split(',')])
        ld = np.array([float(t) for t in tokens[1].split(',')])
        loc = np.array([int(t) for t in tokens[2].split(',')])
        dst = int(tokens[3])
        user_td.append(td)
        user_ld.append(ld)
        user_loc.append(loc)
        user_dst.append(dst)

    if user_td:
        train_user.append(user)
        train_td.append(user_td)
        train_ld.append(user_ld)
        train_loc.append(user_loc)
        train_dst.append(user_dst)
    return train_user, train_td, train_ld, train_loc, train_dst


def print_score(batches, step):
    recall1 = 0.
    recall5 = 0.
    recall10 = 0.
    recall100 = 0.
    recall1000 = 0.
    recall10000 = 0.
    iter_cnt = 0

    for batch in tqdm.tqdm(batches, desc="validation"):
        batch_user, batch_td, batch_ld, batch_loc, batch_dst = batch
        if len(batch_loc) < 3:
            continue
        iter_cnt += 1
        batch_o, target = run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=step)
        print('batch_o: ')
        print('len=', end=' ')
        print(len(batch_o))
        print(batch_o)
        print('batch_user: ')
        print(batch_user)
        recall1 += target in batch_o[:1]
        recall5 += target in batch_o[:5]
        recall10 += target in batch_o[:10]
        recall100 += target in batch_o[:100]
        recall1000 += target in batch_o[:1000]
        recall10000 += target in batch_o[:10000]

    print("recall@1: ", recall1 / iter_cnt)
    print("recall@5: ", recall5 / iter_cnt)
    print("recall@10: ", recall10 / iter_cnt)
    print("recall@100: ", recall100 / iter_cnt)
    print("recall@1000: ", recall1000 / iter_cnt)
    print("recall@10000: ", recall10000 / iter_cnt)


def run(user, td, ld, loc, dst, step):
    optimizer.zero_grad()

    seqlen = len(td)
    user = Variable(torch.from_numpy(np.asarray([user]))).type(ltype)

    rnn_output = h_0
    for idx in range(seqlen - 1):
        td_upper = Variable(torch.from_numpy(np.asarray(up_time - td[idx]))).type(ftype)
        td_lower = Variable(torch.from_numpy(np.asarray(td[idx] - lw_time))).type(ftype)
        ld_upper = Variable(torch.from_numpy(np.asarray(up_dist - ld[idx]))).type(ftype)
        ld_lower = Variable(torch.from_numpy(np.asarray(ld[idx] - lw_dist))).type(ftype)
        location = Variable(torch.from_numpy(np.asarray(loc[idx]))).type(ltype)
        rnn_output = strnn_model(td_upper, td_lower, ld_upper, ld_lower, location, rnn_output)

    td_upper = Variable(torch.from_numpy(np.asarray(up_time - td[-1]))).type(ftype)
    td_lower = Variable(torch.from_numpy(np.asarray(td[-1] - lw_time))).type(ftype)
    ld_upper = Variable(torch.from_numpy(np.asarray(up_dist - ld[-1]))).type(ftype)
    ld_lower = Variable(torch.from_numpy(np.asarray(ld[-1] - lw_dist))).type(ftype)
    location = Variable(torch.from_numpy(np.asarray(loc[-1]))).type(ltype)

    if step > 1:
        return strnn_model.validation(user, td_upper, td_lower, ld_upper, ld_lower, location, dst[-1], rnn_output), dst[-1]

    destination = Variable(torch.from_numpy(np.asarray([dst[-1]]))).type(ltype)
    J = strnn_model.loss(user, td_upper, td_lower, ld_upper, ld_lower, location, destination, rnn_output)  # , neg_lati, neg_longi, neg_loc, step)
    J.backward()
    optimizer.step()
    return J.data.cpu().numpy()


# Parameters
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
train_file = "../data_transfer/strnn/prepro_train_50.txt"
valid_file = "../data_transfer/strnn/prepro_valid_50.txt"
test_file = "../data_transfer/strnn/prepro_test_50.txt"

# Model Hyperparameters
dim = 13    # dimensionality
ww = 360  # winodw width (6h)
up_time = 560632.0  # min
lw_time = 0.
up_dist = 457.335   # km
lw_dist = 0.
reg_lambda = 0.1

# Training Parameters
batch_size = 2
num_epochs = 30
learning_rate = 0.001
momentum = 0.9
evaluate_every = 1
h_0 = Variable(torch.randn(dim, 1), requires_grad=False).type(ftype)

# user_cnt = 32899 #50 #107092#0
# loc_cnt = 1115406 #50 #1280969#0

# Data Preparation
# Load data
print("Loading data...")
# train_user 一维的用户列表
# train_td 一维列表 train_td[0]则是user[0]的td数据 每一个td数据都是list
train_user, train_td, train_ld, train_loc, train_dst = treat_prepro(train_file, step=1)
valid_user, valid_td, valid_ld, valid_loc, valid_dst = treat_prepro(valid_file, step=2)
test_user, test_td, test_ld, test_loc, test_dst = treat_prepro(test_file, step=3)

di = json.load(open('../data_transfer/strnn/config.json', 'r'))
user_cnt = di['user_cnt']
loc_cnt = di['loc_cnt']

print("User/Location: {:d}/{:d}".format(user_cnt, loc_cnt))
print("==================================================================================")

strnn_model = STRNN.STRNNCell(dim, loc_cnt, user_cnt).cuda()
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum, weight_decay=reg_lambda)

for i in range(num_epochs):
    # Training
    total_loss = 0.
    train_batches = list(zip(train_user, train_td, train_ld, train_loc, train_dst))
    for j, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
        #inner_batches = data_loader.inner_iter(train_batch, batch_size)
        # for k, inner_batch in inner_batches:
        batch_user, batch_td, batch_ld, batch_loc, batch_dst = train_batch  # inner_batch)
        if len(batch_loc) < 3:
            continue
        total_loss += run(batch_user, batch_td, batch_ld, batch_loc, batch_dst, step=1)
        # if (j+1) % 2000 == 0:
        #    print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, datetime.datetime.now()
    # Evaluation
    if (i + 1) % evaluate_every == 0:
        print("==================================================================================")
        # print("Evaluation at epoch #{:d}: ".format(i+1)), total_loss/j,
        # datetime.datetime.now()
        valid_batches = list(zip(valid_user, valid_td, valid_ld, valid_loc, valid_dst))
        print_score(valid_batches, step=2)

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = list(zip(test_user, test_td, test_ld, test_loc, test_dst))
print_score(test_batches, step=3)
