import json

origin = open('datasets/small_sample.csv', encoding='utf8')
df = {}

origin.readline()
origin_s = ''
for line in origin.readlines():
    origin_s += line
locs = origin_s.split('(')[1].split(')')[0]
locs = locs.split(',')
for i in range(1, len(locs), 1):
    locs[i] = locs[i][1:]
for i in range(len(locs)):
    locs[i] = (float(locs[i].split(' ')[0]), float(locs[i].split(' ')[1]))
"""print(len(locs))
print(locs)"""

times = origin_s.split('|')[6]
times = times.split(';')
for i in range(len(times)):
    ymd = times[i].split(' ')[0]
    hms = times[i].split(' ')[1]
    year = float(ymd.split('-')[0])
    month = float(ymd.split('-')[1])
    day = float(ymd.split('-')[2])
    hour = float(hms.split(':')[0])
    minute = float(hms.split(':')[1])
    second = float(hms.split(':')[2])
    times[i] = (int(year), int(month), int(day), int(hour), int(minute), int(second))
"""print(times)
print(len(times))"""
df['type'] = 'FeatureCollection'
df['features'] = []
item = {}
item['type'] = 'Feature'
item['properties'] = {}
item['properties']['uid'] = 1
item['properties']['share_extend'] = {}  # some extend info for this traj

geometry = {}
geometry['type'] = 'Polygon'
geometry['coordinates'] = []
for length in range(len(locs)):
    node = {}
    node['location'] = [locs[length][0], locs[length][1]]
    node['time_format'] = ['111111']  # 6bits represent year2second
    # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
    now = times[length]
    time_now = str(now[0])
    for i in range(5):
        time_now += '-'
        time_now += str(now[i + 1])
    node['time'] = [time_now]
    node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
    node['solid_extend'] = {}  # some extend info for this node
    geometry['coordinates'].append(node)

item['geometry'] = geometry
df['features'].append(item)
json.dump(df, open('datasets/small_sample.json', 'w'))
