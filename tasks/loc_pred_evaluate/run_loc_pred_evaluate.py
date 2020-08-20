from evaluate.loc_pred_evaluate import loc_pred_evaluate_model as lpem

if __name__ == '__main__':
    data = '{ "len_pred": 2, ' \
           '"uid1": { ' \
           '"trace_id1":' \
           '{ "loc_true": 1, "loc_pred": [1, 1] }, ' \
           '"trace_id2":' \
           '{ "loc_true": 2, "loc_pred": [2, 3] } ' \
           '},' \
           '"uid2": { ' \
           '"trace_id1":' \
           '{ "loc_true": 3, "loc_pred": [4, 5]}' \
           '}' \
           '}'
    lpt = lpem.LocationPredEvaluate(data, "ACC", 2)
    lpt.run()
