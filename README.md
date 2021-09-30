# TensorflowToolProject
## 基于tensorflow2.6二次开发工程化
### common_util 存放公用数据处理方法
### data 用于存放未处理的数据
### model_util 存放各种模型
### result 存放训练结果
### run_word2vec.py 可以运行word2vec模型
#### 设计思路
1.因为tensorflow1 2 版本差异较大，且百度出来的写法比较繁杂
2.所以依据官方文档tensorflow2.6版本打造，2.6有很多新特性，我也会根据2.6提供的特性来编写
3.第一部分实现，word2vec，实现词向量。参照：https://www.tensorflow.org/tutorials/text/word2vec
4.可以实现中文词向量，具体请参考脚本run_word2vec.py
5.输入语料参照，data下XXX.txt
6.训练参数可以在run_word2vec.py中自己调试
7.中文分词可以采用jieba，分好后按照XXX.txt格式存储