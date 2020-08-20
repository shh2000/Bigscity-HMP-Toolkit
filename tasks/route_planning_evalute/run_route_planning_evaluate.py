from evaluate.route_planning_evalute import route_planning_evaluate_model as rpem

if __name__ == '__main__':
    data = """
    {   "len_pred": 2, 
        "uid1": {
            "trace_id1": {
                "tra_true": [3, 7, 1, 67], 
                "tra_pred": [3, 7, 1, 67]
            }, 
            "trace_id2": {
                "tra_true": [3, 7, 1, 67, 60], 
                "tra_pred": [3, 2, 1, 67, 60]}
            }, 
        "uid2": {
            "trace_id1": {
                "tra_true": [3, 7, 1, 67, 11, 12], 
                "tra_pred": [31, 1, 1, 67, 11, 12]
            }, 
            "trace_id2": {
                "tra_true": [3, 7, 1], 
                "tra_pred": [31, 21, 11]}
            }
    }
    """
    rpt = rpem.RoutePlanningEvaluate(data, mode=1)
    rpt.run()

    data2 = """
        {
            "len_pred": 2, 
            "uid1": { 
                "trace_id1": { 
                    "tra_true": [0, 5, 0, 3, 4, 2, 1, 1, 5, 4], 
                    "tra_pred": [[0, 0, 2, 1, 5], [2, 2, 4, 1, 4], [4, 5, 1, 3, 5], [5, 4, 2, 4, 3], [2, 0, 0, 2, 3], [3, 3, 4, 1, 4], [1, 1, 0, 1, 2], [1, 4, 4, 2, 4], [4, 1, 3, 3, 5], [2, 4, 2, 2, 3]]
                }, 
                "trace_id2": {
                    "tra_true": [0, 5, 0, 3, 4, 2, 1, 1, 5, 4], 
                    "tra_pred": [[0, 0, 2, 1, 5], [2, 2, 4, 1, 4], [4, 5, 1, 3, 5], [5, 4, 2, 4, 3], [2, 0, 0, 2, 3], [3, 3, 4, 1, 4], [1, 1, 0, 1, 2], [1, 4, 4, 2, 4], [4, 1, 3, 3, 5], [2, 4, 2, 2, 3]]
                }
            }, 
            "uid2": {
                "trace_id1": {
                    "tra_true": [0, 5, 0, 3, 4, 2, 1, 1, 5, 4], 
                    "tra_pred": [[0, 0, 2, 1, 5], [2, 2, 4, 1, 4], [4, 5, 1, 3, 5], [5, 4, 2, 4, 3], [2, 0, 0, 2, 3], [3, 3, 4, 1, 4], [1, 1, 0, 1, 2], [1, 4, 4, 2, 4], [4, 1, 3, 3, 5], [2, 4, 2, 2, 3]]
                }, 
                "trace_id2": {
                    "tra_true": [0, 5, 0, 3, 4, 2, 1, 1, 5, 4], 
                    "tra_pred": [[0, 0, 2, 1, 5], [2, 2, 4, 1, 4], [4, 5, 1, 3, 5], [5, 4, 2, 4, 3], [2, 0, 0, 2, 3], [3, 3, 4, 1, 4], [1, 1, 0, 1, 2], [1, 4, 4, 2, 4], [4, 1, 3, 3, 5], [2, 4, 2, 2, 3]]
                }
            }
        }
    """
    rpt = rpem.RoutePlanningEvaluate(data2, mode=2, k=3)
    rpt.run()

