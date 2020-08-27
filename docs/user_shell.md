## 启动方式

* cd users
* python parse_arg.py []

## 命令行参数使用方法

* help

![image-20200827164106938](C:\Users\shh2000\AppData\Roaming\Typora\typora-user-images\image-20200827164106938.png)

* -t run

  选择任务类型，有查看环境中存在的数据集、运行所有的预置模型、加入新模型、加入新数据集等4个操作

* -data

  加入新数据集时必须有，为字符串。具体的存放要求还没弄好，要和handler联合。初步计划放在data_extract/datasets下

* -model -model_type

  加入新模型时必须有，为字符串。这不是一个单纯的模型，而是要写好一个符合格式的main函数，要接受字符串格式的数据集作为输入，同时将结果按照规定的格式（预测/规划）保存到指定位置。这个位置后面联合deepmove的结果存放位置，同时在evaluate系列handler里面从这里读东西，和模型本身分离