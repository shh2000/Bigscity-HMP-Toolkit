from data_transfer.deep_move_transfer import deepMoveTransfer
import json
import os

def gen_data(model_name, dataset_name, *args):
    # TODO: 判断 model_name 与 dataset 的方法应该还有待改进，比如硬编码？
    # cache with parameters
    parameters_str = ''
    for i in args:
        parameters_str += '_' + str(i)
    cache_file_name = '{}_{}{}.json'.format(model_name, dataset_name, parameters_str)
    if model_name == 'deepMove' or model_name == 'simpleRNN':
        # cache 的话会把 num 的 key 转换成 str 的 key 导致出错
        if os.path.exists('../data_transfer/datasets/{}'.format(cache_file_name)):
            with open('../data_transfer/datasets/{}'.format(cache_file_name), 'r') as f:
                data = json.load(f)
                return data
        with open('../data_extract/datasets/traj_' + dataset_name + '.json', 'r') as f:
            data = json.load(f)
            data_transformed = deepMoveTransfer(data, args[0], args[1], args[2])
            cache = open('../data_transfer/datasets/{}'.format(cache_file_name), 'w')
            json.dump(data_transformed, cache)
            cache.close()
            return data_transformed
    else:
        # not implement TODO: throw Error
        return None