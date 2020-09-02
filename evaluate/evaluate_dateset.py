# 评估选取不同的 transfer 参数，不同数据集生成出来的模型输入数据的以下指标：
# 用户平均轨迹数，轨迹平均长度，总用户数，总轨迹条数
import sys
import json
sys.path.append('..')
from data_transfer import deepMoveTransfer

# 检查数据集正确性
# min_tim = 100
# max_tim = 0
# for u in data_neural:
#     for session in data_neural[u]['sessions']:
#         for node in data_neural[u]['sessions'][session]:
#             if min_tim > node[1]:
#                 min_tim = node[1]
#                 print(u, session, node)
#             if max_tim < node[1]:
#                 max_tim = node[1]
#                 print(u, session, node)

time_length = 12
min_session_len = 5
min_sessions = 2
dataset_name = 'traj_gowalla'
with open('../data_extract/datasets/{}.json'.format(dataset_name), 'r') as f:
    data = json.load(f)
    data_transformed = deepMoveTransfer(data, min_session_len, min_sessions, time_length)
    print('finish')
    user_cnt = data_transformed['uid_size']
    session_cnt = 0
    session_len = 0
    for user in data_transformed['data_neural']:
        session_cnt += len(data_transformed['data_neural'][user]['sessions'])
        for session in data_transformed['data_neural'][user]['sessions']:
            session_len += len(data_transformed['data_neural'][user]['sessions'][session])
    avg_user_session = session_cnt / user_cnt
    avg_session_len = session_len / session_cnt
    print('{} datasets'.format(dataset_name))
    print('min_session_len: ', min_session_len)
    print('min_sessions: ', min_sessions)
    print('time_length: ', time_length)
    print('total user: ', user_cnt)
    print('total session: ', session_cnt)
    print('avg_user_session: ', avg_user_session)
    print('avg_session_len: ', avg_session_len)