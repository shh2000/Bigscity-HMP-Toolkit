import json
from random import randint, random
import datetime


def gen_long_lati():
    o = random()
    o = o - 0.5
    o *= 360
    return o


def gen_traj():
    num = 10
    df = {}
    df['type'] = 'FeatureCollection'
    df['features'] = []
    for index in range(num):
        item = {}
        item['type'] = 'Feature'
        item['properties'] = {}
        item['properties']['uid'] = index + 1
        item['properties']['share_extend'] = {}  # some extend info for this traj

        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = []
        for length in range(randint(4, 15)):
            node = {}
            node['location'] = [gen_long_lati(), gen_long_lati()]
            node['time_format'] = ['111111']  # 6bits represent year2second
            # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
            node['time'] = ['2020-02-01-13-59-32']
            node['index'] = length  # if time_format is not 111111, we must use index to determine the sequence
            node['solid_extend'] = {}  # some extend info for this node
            geometry['coordinates'].append(node)

        item['geometry'] = geometry
        df['features'].append(item)
    json.dump(df, open('traj_sample.json','w'))

gen_traj()
