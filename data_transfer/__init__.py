from data_transfer.deep_move_transfer import deepMoveTransfer
import json

def gen_data(model_name, dataset_name):
    # TODO: 判断 model_name 与 dataset 的方法应该还有待改进，比如硬编码？
    if model_name == 'deepMove':
        with open('../data_extract/' + dataset_name + '.json', 'r') as f:
            data = json.load(f)
            data_transformed = deepMoveTransfer(data)
            return data_transformed
    else:
        # not implement TODO: throw Error
        return None