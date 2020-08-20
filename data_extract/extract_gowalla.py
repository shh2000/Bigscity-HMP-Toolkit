import json

origin = open('datasets/Gowalla_totalCheckins.txt')
lines = origin.readlines()
#lines = lines[0:456860]
id2lines = {}

last_id = 0
last_start_line = 0
for i in range(len(lines)):
    id = lines[i].split('\t')[0]
    id = int(id)
    # print(id)
    if id != last_id:
        id2lines[last_id] = (last_start_line, i)
        last_id = id
        last_start_line = i
print('finish pre')
df = {}
df['type'] = 'FeatureCollection'
df['features'] = []
last_line = 0
for id in id2lines.keys():
    if id2lines[id][0] > last_line + 300000:
        print(id2lines[id][0])
        last_line = id2lines[id][0]
    item = {}
    item['type'] = 'Feature'
    item['properties'] = {}
    item['properties']['uid'] = id
    item['properties']['share_extend'] = {}  # some extend info for this traj

    geometry = {}
    geometry['type'] = 'Polygon'
    geometry['coordinates'] = []
    for length in range(id2lines[id][0], id2lines[id][1], 1):
        node = {}
        line = lines[length]
        long = float(line.split('\t')[2])
        lati = float(line.split('\t')[3])
        loc_id = int(line.replace('\n', '').split('\t')[-1])
        node['location'] = [long, lati]
        node['time_format'] = ['111111']  # 6bits represent year2second
        # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
        time = line.split('\t')[1]
        time = time.replace('T', '-')
        time = time.replace(':', '-')
        time = time.replace('Z', '')
        node['time'] = [time]
        node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
        node['solid_extend'] = {'loc_id': loc_id}  # some extend info for this node
        geometry['coordinates'].append(node)

    item['geometry'] = geometry
    df['features'].append(item)
json.dump(df, open('datasets/traj_gowalla.json', 'w'))
