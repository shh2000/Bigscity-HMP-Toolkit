from data_transfer.deep_move_transfer import deepMoveTransfer
import json
import os

def gen_data(model_name, dataset_name):
    # TODO: 判断 model_name 与 dataset 的方法应该还有待改进，比如硬编码？
    if model_name == 'deepMove':
        # cache 的话会把 num 的 key 转换成 str 的 key 导致出错
        if os.path.exists('../data_transfer/datasets/deepMove_{}.json'.format(dataset_name)):
            with open('../data_transfer/datasets/deepMove_{}.json'.format(dataset_name), 'r') as f:
                data = json.load(f)
                return data
        with open('../data_extract/datasets/' + dataset_name + '.json', 'r') as f:
            data = json.load(f)
            data_transformed = deepMoveTransfer(data)
            cache = open('../data_transfer/datasets/deepMove_{}.json'.format(dataset_name), 'w')
            json.dump(data_transformed, cache)
            cache.close()
            return data_transformed
    else:
        # not implement TODO: throw Error
        return None