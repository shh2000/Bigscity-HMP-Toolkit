from evaluate import loc_pred_evaluate_model as lpem

if __name__ == '__main__':
    data = '{' \
           '"uid1": { ' \
           '"trace_id1":' \
           '{ "loc_true": [1], "loc_pred": [[0.01, 0.91, 0.8]] }, ' \
           '"trace_id2":' \
           '{ "loc_true": [2], "loc_pred": [[0.2, 0.13, 0.08]] } ' \
           '},' \
           '"uid2": { ' \
           '"trace_id1":' \
           '{ "loc_true": [0], "loc_pred": [[0.4, 0.5, 0.7]] }' \
           '}' \
           '}'
    lpt = lpem.LocationPredEvaluate(data, "DeepMove", "ACC", 1, 2)
    lpt.run()
