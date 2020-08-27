>此文件负责说明 trajectory prediction 任务开放给用户自定义的一些参数

#### transfer 相关参数

* min_session_len：轨迹最短长度。
  * 限制：不得小于 2。因为如果轨迹中只有一个点，是没有办法做预测的，即不存在下一跳。
* min_sessions：用户最少轨迹数量。
  * 限制：不得小于 2。如果一个用户轨迹数小于 2，就没有办法找到历史轨迹数据，会发生错误。
* time_length：轨迹间时间间隔。
  * 限制：需为 12 的整数倍，单位为小时。

#### task 相关参数

* model_name: 指定使用模型的名称。
  * 目前仅支持 SimpleRNN、DeepMove。
* datasets: 指定使用的数据集。
  * 目前支持 traj_foursquare、traj_gowalla。但是 traj_gowalla 存在性能问题，在数据 transfer 阶段会很慢。
