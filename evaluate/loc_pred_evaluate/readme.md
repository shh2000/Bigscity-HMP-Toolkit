### 位置预测评估模型

@params

1. data：传入评估模型的用户轨迹数据及预测数据，要求满足 json 格式数据，详见该目录下的 data.md 文档

2. mode：评估选项/指标
    1. ACC, 计算预测准确度（Accuracy）
    2. MSE：均方误差（Mean Square Error）
    3. MAE：平均绝对误差（Mean Absolute Error）
    4. RMSE：均方根误差（Root Mean Square Error）
    5. MAPE：平均绝对百分比误差（Mean Absolute Percentage Error）
    6. MARE：平均绝对和相对误差（Mean Absolute Relative Error）

3. k：对应 top-k 评估方法的k参数，默认为1
    1. top-k：check whether the ground-truth location v appears in the top-k candidate locations
    2. top-k：真实位置在前k个预测位置中准确的准确度
    3. e.g. 某个数据集上所有模型整体预测准确的偏低的情况，可采用top-5
