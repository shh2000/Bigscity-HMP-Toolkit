# 位置预测评估

## 文件组织格式

|-- evaluate -- loc_pred_evlauate_model.py 评估模型

​					  -- loc_pred_evaluate_data.py	DataSet结构

|-- docs -- loc_pred_evaluate.md 					位置预测评估任务的说明文档

|-- tasks -- run_loc_pred_evaluate					运行样例

## 使用说明

### 调用方法

当需要进行位置预测评估时：

1. 引入 `/evaluate` 目录下的 `loc_pred_evaluate_model.py` 文件。
2. 创建 `LocationPredEvaluate` 类。
3. 调用 `run` 方法。
4. 可参考 `/tasks/run_loc_pred_evaluate.py` 文件。

```python
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
```

### 参数含义

以 `lpt = lpem.LocationPredEvaluate(data, "DeepMove", "ACC", 1, 2)` 为例：

data：满足位置预测评估任务的 json 格式的数据，用于评测。(必传)

datatype：数据类型，若是 DeepMove 模型产生的数据，则此参数为 "DeepMove".（选传）

mode：选择对该模型采用的评估方法，默认为 'ACC'，其他的评估方法可见该文档末尾。（选传）

k：对应 top-k ACC 评估方法中的 k 参数，默认为1.（选传）

len_pred：对每个真实位置预测产生的预测位置数量，对应输入数据中的 `loc_pred` 的列的个数，即 `loc_pred` 的每个列表元素的大小，默认为1.（选传）

## 数据格式说明

附：关于位置的表示，参考DeepMove模型对所有的位置进行独热编码，每个位置具有一个编号。

```
{
	uid1: {
		trace_id1: {
			loc_true: 一个list类型列表，列表中元素代表真实位置(编号)
			loc_pred: 一个list类型列表，列表中的元素又是一个list类型列表，代表 [模型预测出的位置(编号)]。按照预测可能性的大小排列，最有可能的位置排在第1位；也可以是列表中元素代表 [模型预测出的位置(编号)的置信度]，比如DeepMove模型的输出。
		},
		trace_id2: {
			...
		},
		...
	},
	uid2: {
		...
	},
	...
}
```

样例：

```
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
```

## 评估方法

mode：评估选项/指标

1. ACC, 计算预测准确度（Accuracy）
2. MSE：均方误差（Mean Square Error）
3. MAE：平均绝对误差（Mean Absolute Error）
4. RMSE：均方根误差（Root Mean Square Error）
5. MAPE：平均绝对百分比误差（Mean Absolute Percentage Error）
6. MARE：平均绝对和相对误差（Mean Absolute Relative Error）
7. top-k：check whether the ground-truth location v appears in the top-k candidate locations，即真实位置在前k个预测位置中准确的准确度，若某个数据集上所有模型整体预测准确的偏低的情况，可采用top-5。