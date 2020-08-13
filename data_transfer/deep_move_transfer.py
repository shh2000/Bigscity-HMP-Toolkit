import geohash2
import json
from datetime import datetime
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

def deepMoveTransfer(data):
    '''
    data: raw data which obey the trajectory data format
    {
        uid: {
            session_id: {
                loc: [],
                tim: []
            }, 按照 48 h 的时间间隔将一个用户的 trace 划分成多段 session
            ...
        }
    }
    '''
    features = data['features']
    # 因为 DeepMove 将对 loc 进行了 labelEncode 所以需要先获得 loc 的全集
    loc_set = []
    data_transformed = {}
    for feature in features:
        uid = feature['properties']['uid']
        user_data = {}
        traj_data = feature['geometry']['coordinates']
        session_id = 1
        session = {
            'loc': [],
            'tim': []
        }
        if len(traj_data) == 0:
            # TODO: shouldn't happen this
            continue
        start_time = parseTime(traj_data[0]['time'], traj_data[0]['time_format'])
        for index, node in enumerate(traj_data):
            loc_hash = encodeLoc(node['location'])
            loc_set.append(loc_hash)
            if index == 0:
                session['loc'].append(loc_hash)
                session['tim'].append(start_time.hour) # time encode from 0 ~ 47
            else:
                now_time = parseTime(node['time'], node['time_format'])
                if now_time.day - start_time.day < 2:
                    # stay in the same session
                    session['loc'].append(loc_hash)
                    session['tim'].append(now_time.hour if now_time.day - start_time.day == 0 else now_time.hour + 24)
                else:
                    # new session will be created
                    user_data[session_id] = session
                    # clear session and add session_id
                    session_id += 1
                    session = {
                        'loc': [],
                        'tim': []
                    }
                    start_time = now_time
                    session['loc'].append(loc_hash)
                    session['tim'].append(start_time.hour)
        
        user_data[session_id] = session
        data_transformed[uid] = user_data

    # label encode
    encoder = LabelEncoder()
    encoder.fit(loc_set)

    # do loc labelEncoder
    for user in data_transformed.keys():
        for session in data_transformed[user].keys():
            data_transformed[user][session]['loc'] = encoder.transform(data_transformed[user][session]['loc']).tolist()
    
    return data_transformed

# for loc test
with open('../data_extract/traj_sample.json', 'r') as f:
    data = json.load(f)
    data_transformed = deepMoveTransfer(data)
    res = open('./deep_move_traj_sample.json', 'w')
    json.dump(data_transformed, res)
    res.close()