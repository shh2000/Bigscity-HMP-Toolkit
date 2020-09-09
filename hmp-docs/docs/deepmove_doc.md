## train

由于很多路径是相对路径，目前只能 cd 到 tasks 里面去 run。

```
cd tasks
python train_deep_move.py
```

## evaluate

这部分添加加载预训练好的模型的代码，现在是空的未训练的模型来评估。

```
cd tasks
python evaluate_deep_move.py
```

## 调用逻辑

先调用 data_transfer 下的 gen_data 生成数据，在借助 utils 里面的 gen_history 计算历史数据以及切分训练集、测试集，最后加载 model 里面的模型跑就完事了。

