import geohash2
import json
import math
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

def encodeLoc(loc):
    return geohash2.encode(loc[0], loc[1]) # loc[0]: latitude, loc[1]: longtitude 

def parseTime(time, time_format):
    '''
    parse to datetime
    '''
    # only implement 111111
    if time_format[0] == '111111':
        return datetime.strptime(time[0], '%Y-%m-%d-%H-%M-%S') # TODO: check if this is UTC Time ?

def calculateBaseTime(start_time, base_zero):
    if base_zero:
        return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
    else:
        # time length = 12
        if start_time.hour < 12:
            return start_time - timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)
        else:
            return start_time - timedelta(hours=start_time.hour - 12, minutes=start_time.minute, seconds=start_time.second,microseconds=start_time.microsecond)

def calculateTimeOff(now_time, base_time):
    # 先将 now 按小时对齐
    now_time = now_time - timedelta(minutes=now_time.minute, seconds=now_time.second)
    delta = now_time - base_time
    return delta.days * 24 + delta.seconds / 3600

def deepMoveTransfer(data, min_session_len = 5, min_sessions = 2, time_length = 72):
    '''
    data: raw data which obey the trajectory data format
    min_session_len: the min number of nodes in a session
    min_sessions: the min number of sessions for a user
    time_length: use for cut raw trajectory into sessions (需为 12 的整数倍)
    output:
    {
        uid: {
            sessions: {
                session_id: [
                    [loc, tim],
                    [loc, tim]
                ],
                ....
            }, 按照 time_length 的时间间隔将一个用户的 trace 划分成多段 session
            train: [0, 1, 2],
            test: [3, 4] 按照一定比例，划分 train 与 test 合集。目前暂定是后 25% 的 session 作为 test
        }
    }
    '''
    base_zero = time_length > 12 # 只对以半天为间隔的做特殊处理
    features = data['features']
    # 因为 DeepMove 将对 loc 进行了 labelEncode 所以需要先获得 loc 的全集
    loc_set = []
    data_transformed = {}
    for feature in features:
        uid = feature['properties']['uid']
        sessions = {}
        traj_data = feature['geometry']['coordinates']
        session_id = 1
        session = {
            'loc': [],
            'tim': []
        }
        if len(traj_data) == 0:
            # TODO: shouldn't happen this, throw error ?
            continue
        start_time = parseTime(traj_data[0]['time'], traj_data[0]['time_format'])
        base_time = calculateBaseTime(start_time, base_zero)
        for index, node in enumerate(traj_data):
            loc_hash = encodeLoc(node['location'])
            loc_set.append(loc_hash)
            if index == 0:
                session['loc'].append(loc_hash)
                session['tim'].append(start_time.hour - base_time.hour) # time encode from 0 ~ time_length
            else:
                now_time = parseTime(node['time'], node['time_format'])
                time_off = calculateTimeOff(now_time, base_time)
                if time_off < time_length and time_off >= 0: # should not be 乱序
                    # stay in the same session
                    session['loc'].append(loc_hash)
                    session['tim'].append(time_off)
                else:
                    if len(session['loc']) >= min_session_len:
                        # session less than 2 point should be filtered, because this will cause target be empty
                        # new session will be created
                        sessions[str(session_id)] = session
                        # clear session and add session_id
                        session_id += 1
                    session = {
                        'loc': [],
                        'tim': []
                    }
                    start_time = now_time
                    base_time = calculateBaseTime(start_time, base_zero)
                    session['loc'].append(loc_hash)
                    session['tim'].append(start_time.hour - base_time.hour)
        if len(session['loc']) >= min_session_len:
            sessions[str(session_id)] = session
        else:
            session_id -= 1
        # TODO: there will be some trouble with only one session user
        if len(sessions) >= min_sessions:
            data_transformed[str(uid)] = {}
            data_transformed[str(uid)]['sessions'] = sessions
            # 25% session will be test session
            split_num = math.ceil(session_id*0.6) + 1
            data_transformed[str(uid)]['train'] = [str(i) for i in range(1, split_num)]
            if split_num < session_id:
                data_transformed[str(uid)]['test'] = [str(i) for i in range(split_num, session_id + 1)]
            else:
                data_transformed[str(uid)]['test'] = []
    # label encode
    print('start encode')
    print('loc size ', len(loc_set))
    encoder = LabelEncoder()
    encoder.fit(loc_set)
    print('finish encode')

    # do loc labelEncoder
    verbose = 100
    cnt = 0
    total_len = len(data_transformed)
    for user in data_transformed.keys():
        for session in data_transformed[user]['sessions'].keys():
            temp = []
            # TODO: any more effecient way to do this ?
            length = len(data_transformed[user]['sessions'][session]['tim'])
            loc_tmp = encoder.transform(data_transformed[user]['sessions'][session]['loc']).reshape(length, 1).astype(int)
            tim_tmp = np.array(data_transformed[user]['sessions'][session]['tim']).reshape(length, 1).astype(int)
            data_transformed[user]['sessions'][session] = np.hstack((loc_tmp, tim_tmp)).tolist()
        cnt += 1
        if cnt % verbose == 0:
            print('data encode finish: {}/{}'.format(cnt, total_len))

    res = {
        'data_neural': data_transformed,
        'loc_size': encoder.classes_.shape[0],
        'uid_size': len(data_transformed)
    }

    return res

# for loc test
if __name__ == "__main__":
    f = open('../data_extract/datasets/traj_foursquare.json', 'r')
    data = json.load(f)
    f.close()
    data_transformed = deepMoveTransfer(data)