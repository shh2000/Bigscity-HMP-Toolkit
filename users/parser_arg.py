import json
import argparse

import os


def run_model_on_dataset(model, dataset):
    if model == 'deepMove' and dataset == 'foursquare':
        os.system(r'cd ..\tasks && python train_traj_prediction.py {} {}'.format(model, dataset))
        return True
    else:
        print('You want to run {} on {}, no this model or datasets!'.format(model, dataset))
        return False


def evaluate_model_on_dataset(model, dataset, model_type):
    if model == 'deepmove' and dataset == 'foursquare' and model_type == 'predict':
        os.system(r'cd ..\tasks && python evaluate_traj_prediction.py {} {}'.format(model, dataset))
    else:
        print('You want to evaluate {}-type model {}, no this model!'.format(model, model_type))


_info = json.load(open('preset.json'))
_datasets = _info['datasets']
_models = _info['models']

SUCCEED = 0
INVALID_ARG = -1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--app_type", required=True, default="new_data", help="new_data/new_model/run/show")
    parser.add_argument("-data", "--my_dataset", default="all", help="input new_data's name")
    parser.add_argument("-model", "--my_model", default="all", help="input new_model's name")
    parser.add_argument("-model_type", "--my_model_type", default="all", help="input new_model's type(predict/plan)")
    arg = parser.parse_args()
    return arg


def check_data(arg):
    if arg.my_dataset is None:
        return INVALID_ARG
    return SUCCEED


def check_model(arg):
    if arg.my_model is None:
        return INVALID_ARG
    return SUCCEED


def check_model_type(arg):
    if arg.my_model_type is None:
        return INVALID_ARG
    if arg.my_model_type == 'predict' or arg.my_model_type == 'plan':
        return SUCCEED
    return INVALID_ARG


if __name__ == '__main__':
    arg = parse()
    datas2run = []
    models2run = {}
    if arg.app_type == 'show':
        print('Datasets: ', _datasets)
        print('Predict Models: ', _models['predict'])
        print('Plan Models: ', _models['plan'])
        exit(0)
    else:
        print('App created succeed!')
        print('Your task is {}'.format(arg.app_type))
        if arg.app_type == 'run':
            models2run = _models
            datas2run = _datasets
        elif arg.app_type == 'new_data':
            r = check_data(arg)
            if r != SUCCEED:
                print('Invalid new dataset!')
                exit(r)
            print('Your new Dataset {} has been successfully identified!'.format(arg.my_dataset))
            tmp = _datasets
            tmp.append(arg.my_dataset)
            models2run = _models
            datas2run = tmp
        elif arg.app_type == 'new_model':
            r = check_model(arg)
            if r != SUCCEED:
                print('Invalid new model!')
                exit(r)
            print('Your new Model {} has been successfully identified!'.format(arg.my_model))
            r = check_model_type(arg)
            if r != SUCCEED:
                print('Invalid model type!')
                exit(r)
            tmp = _models
            if arg.my_model_type == 'predict':
                tmp['predict'].append(arg.my_model)
            elif arg.my_model_type == 'plan':
                tmp['plan'].append(arg.my_model)
            models2run = tmp
            datas2run = _datasets
        else:
            print('Invalid app type!')
            exit(0)

        print('Datasets ready to run: ', datas2run)
        print('Models ready to run: ', models2run)
        print('Begin calculating:')

        for model in models2run['predict']:
            for dataset in datas2run:
                if run_model_on_dataset(model, dataset):
                    evaluate_model_on_dataset(model, dataset, 'predict')

        for model in models2run['plan']:
            for dataset in datas2run:
                if run_model_on_dataset(model, dataset):
                    evaluate_model_on_dataset(model, dataset, 'plan')

        print('Calculate finished! Begin generating report:')

        exit(0)
