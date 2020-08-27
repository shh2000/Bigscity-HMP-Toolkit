import json

monthkey = {'Jan': '01', 'Feb': '02', 'Mar': '03'
    , 'Apr': '04', 'May': '05', 'Jun': '06'
    , 'Jul': '07', 'Aug': '08', 'Sep': '09'
    , 'Oct': '10', 'Nov': '11', 'Dec': '12'}

origin = open('datasets/dataset_TSMC2014_TKY.csv', encoding='utf8')
origin.readline()
lines = origin.readlines()
id2lines = {}
cnt = 0
for line in lines:
    info = line.replace('\n', '').split(',')
    id = info[0]
    if id not in id2lines.keys():
        id2lines[id] = []
    id2lines[id].append(cnt)
    cnt += 1
# print(id2lines)
df = {}
df['type'] = 'FeatureCollection'
df['features'] = []
for id in id2lines.keys():
    item = {}
    item['type'] = 'Feature'
    item['properties'] = {}
    item['properties']['uid'] = id
    item['properties']['share_extend'] = {}  # some extend info for this traj

    geometry = {}
    geometry['type'] = 'Polygon'
    geometry['coordinates'] = []
    for length in id2lines[id]:
        node = {}
        line = lines[length]
        long = float(line.split(',')[5])
        lati = float(line.split(',')[4])
        venue_id = line.split(',')[1]
        node['location'] = [long, lati]
        node['time_format'] = ['111111']  # 6bits represent year2second
        # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
        time = line.split(',')[-1].replace('\n', '')
        day = time.split(' ')[2]
        year = time.split(' ')[-1]
        hms = time.split(' ')[3].replace(':', '-')
        month = time.split(' ')[1]
        month = monthkey[month]
        time = year + '-' + month + '-' + day + '-' + hms
        node['time'] = [time]
        node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
        node['solid_extend'] = {'venue_id': venue_id}  # some extend info for this node
        geometry['coordinates'].append(node)

    item['geometry'] = geometry
    df['features'].append(item)
json.dump(df, open('datasets/traj_foursquare-tky.json', 'w'))
