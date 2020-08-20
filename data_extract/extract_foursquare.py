import json

origin = open('datasets/checkins.dat', encoding='utf8')
origin.readline()
origin.readline()
lines = origin.readlines()
id2lines = {}
cnt = 0
for line in lines:
    if '|' not in line:
        break
    info = line.replace('\n', '').split('|')
    id = info[1].replace(' ', '')
    long = info[3]
    if '.' not in long:
        cnt += 1
        continue
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
        long = float(line.split('|')[3].replace(' ', ''))
        lati = float(line.split('|')[4].replace(' ', ''))
        venue_id = int(line.split('|')[2].replace(' ', ''))
        node['location'] = [long, lati]
        node['time_format'] = ['111111']  # 6bits represent year2second
        # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
        time = line.split('|')[-1][1:]
        time = time.replace(' ', '-')
        time = time.replace(':', '-')
        time = time.replace('\n', '')
        node['time'] = [time]
        node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
        node['solid_extend'] = {'venue_id': venue_id}  # some extend info for this node
        geometry['coordinates'].append(node)

    item['geometry'] = geometry
    df['features'].append(item)
json.dump(df, open('datasets/traj_foursquare.json', 'w'))
