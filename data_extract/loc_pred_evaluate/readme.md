### 位置预测评估 DataSet

传入 json 格式的数据集，方便使用 pytorch 工具对数据进行随机化或分片化处理。

**注意**：相比于 evaluate 文件夹下的传入数据，少了 len_pred 字段，即该DataSet获得的只包括用户的轨迹数据与预测数据，并不care预测模型预测的位置数量 