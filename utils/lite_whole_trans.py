import json


def lite2whole(litepath, wholepath):
    lite = json.load(open(litepath))
    whole = {}
    whole['type'] = 'FeatureCollection'
    whole['features'] = []
    for liteitem in lite['features']:
        item = {}
        item['type'] = 'Feature'
        item['properties'] = {}
        item['properties']['uid'] = liteitem['p']['id']
        item['properties']['share_extend'] = {}  # some extend info for this traj

        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = []
        for litenode in liteitem['g']['c']:
            node = {}
            node['location'] = litenode['l']
            node['time_format'] = ['111111']  # 6bits represent year2second
            # if format == 000111, time would be 13hour, 59min, 32sec, without year, month, day
            node['time'] = litenode['t']
            node['index'] = -1  # if time_format is not 111111, we must use index to determine the sequence
            node['solid_extend'] = litenode['ex']  # some extend info for this node
            geometry['coordinates'].append(node)

        item['geometry'] = geometry
        whole['features'].append(item)
    json.dump(whole, open(wholepath, 'w'))


def whole2lite(wholepath, litepath):
    whole = json.load(open(wholepath))
    lite = {}
    lite['type'] = 'FeatureCollection'
    lite['features'] = []
    for wholeitem in whole['features']:
        item = {}
        item['t'] = 'Feature'
        item['p'] = {}
        item['p']['id'] = wholeitem['properties']['uid']

        geometry = {}
        geometry['t'] = 'Polygon'
        geometry['c'] = []
        for wholenode in wholeitem['geometry']['coordinates']:
            node = {}
            node['l'] = wholenode['location']
            node['t'] = wholenode['time']
            node['ex'] = wholenode['solid_extend']  # some extend info for this node
            geometry['c'].append(node)

        item['g'] = geometry
        lite['features'].append(item)
    json.dump(lite, open(litepath, 'w'))


lite2whole('../data_extract/traj_sample-lite.json', '../data_extract/datasets/traj_sample.json')
whole2lite('../data_extract/traj_sample.json', '../data_extract/datasets/traj_sample-lite.json')

whole2lite('../data_extract/datasets/traj_gowalla.json','../data_extract/datasets/traj_gowalla-lite.json')
whole2lite('../data_extract/datasets/traj_foursquare.json','../data_extract/datasets/traj_foursquare-lite.json')
