from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
import sys
sys.path.append('..')
from utils import encodeLoc, parseTime, calculateTimeOff

def FPMCTransfer(data, min_session_len = 2, min_sessions = 1, time_length = 72):
    '''
    min_session_len: > 2
    min_sessions: >= 1, all ok. just keep the same as deepMove
    time_length: keep the same as deepMoveTransfer
    return:{
        data_list: [
            (uid, target, pre-position: list)
        ],
        n_user,
        loc_set
    }
        
    '''
    # 直接硬切吧，不找 base 了
    features = data['features']
    loc_set = []
    data_list = []
    uid = 0
    for feature in features:
        # uid = feature['properties']['uid'] # for the dataset's uid don't start from 0, so just count from 0.
        session = []
        traj_data = feature['geometry']['coordinates']
        if len(traj_data) == 0:
            continue
        base_time = parseTime(traj_data[0]['time'], traj_data[0]['time_format'])
        for index, node in enumerate(traj_data):
            loc_hash = encodeLoc(node['location'])
            loc_set.append(loc_hash)
            if index == 0:
                session.append(loc_hash)
            else:
                now_time = parseTime(node['time'], node['time_format'])
                time_off = calculateTimeOff(now_time, base_time)
                if time_off < time_length and time_off >= 0:
                    session.append(loc_hash)
                else:
                    session = np.unique(session).tolist() # TODO: 这里去重是否合适
                    if len(session) >= min_session_len:
                        data_list.append((uid, session))
                    session = []
                    base_time = now_time
                    session.append(loc_hash)
        uid += 1
    print('start encode')
    print('loc size', len(loc_set))
    encoder = LabelEncoder()
    encoder.fit(loc_set)
    loc_set = [i for i in range(len(encoder.classes_))]
    print('finish encode')
    verbose = 100
    total = len(data_list)
    for i in range(len(data_list)):
        length = len(data_list[i][1])
        loc_temp = encoder.transform(data_list[i][1]).tolist()
        data_list[i] = (data_list[i][0], loc_temp[-1], loc_temp)
        if i % verbose == 0:
            print('finish {}/{}'.format(i, total))
    return {
        'data_list': data_list,
        'n_user': uid,
        'loc_set': loc_set
    }

# for local test
if __name__ == "__main__":
    f = open('../data_extract/datasets/traj_foursquare.json')
    data = json.load(f)
