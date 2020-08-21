from evaluate.loc_pred_evaluate import loc_pred_evaluate_model as lpem
import json

if __name__ == '__main__':
    data = '{' \
           '"uid1": { ' \
           '"trace_id1":' \
           '{ "loc_true": [1], "loc_pred": [[0.01, 0.1]] }, ' \
           '"trace_id2":' \
           '{ "loc_true": [2], "loc_pred": [[0.2, 0.13]] } ' \
           '},' \
           '"uid2": { ' \
           '"trace_id1":' \
           '{ "loc_true": [3], "loc_pred": [[4, 5]] }' \
           '}' \
           '}'
    lpt = lpem.LocationPredEvaluate(data, "DeepMove", "ACC", 1, 2)
    lpt.run()
