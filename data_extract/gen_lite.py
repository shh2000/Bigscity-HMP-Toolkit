import json
from random import randint, random
import datetime


def gen_long_lati():
    o = random()
    o = o - 0.5
    o *= 360
    return o


def gen_traj_lite():
    num = 10
    df = {}
    df['type'] = 'FeatureCollection'
    df['features'] = []
    for index in range(num):
        item = {}
        item['t'] = 'Feature'
        item['p'] = {}
        item['p']['id'] = index + 1

        geometry = {}
        geometry['t'] = 'Polygon'
        geometry['c'] = []
        for length in range(randint(4, 15)):
            node = {}
            node['l'] = [gen_long_lati(), gen_long_lati()]
            node['t'] = ['2020-02-01-13-59-32']
            node['ex'] = {}  # some extend info for this node
            geometry['c'].append(node)

        item['g'] = geometry
        df['features'].append(item)
    json.dump(df, open('traj_sample-lite.json','w'))

gen_traj_lite()
