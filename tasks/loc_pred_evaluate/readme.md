### 位置预测评估

使用方法

1. 引入 evaluate 目录下的 loc_pred_evaluate_model

2. 实例化一个评估模型，共有3个(可选)参数
   1. data，满足位置预测任务评估 json 格式的数据(必传)
   2. mode，选择对位置预测模型的评估方法，默认为ACC
   3. k，对应 top-k ACC 评估方法中的k参数，默认为1

3. 调用该模型的 run 方法运行