## Sampling Matters in Deep Embedding Learning

基于triplet三元组，根据图像之间的相似概率进行采样选择positive和negtive样本  

## 兼容

Python 3.6  
Torch 1.1.0  
Torchvison 0.3.0



## 更新

| Date       | Update     |
| ---------- | ---------- |
| 2019-05-21 | 第一次上传 |

## 

## 数据

如果想训练你自己的数据，那么你的数据结构需要如下（根目录文件夹格式）

- animals
  - birds
    - birds_1.jpg
    - ...
  - fish
    - fish_1.jpg
    - ...
- holidays
  - christmas
    - christmas_1.jpg
    - ...



## 训练

```
# 生成训练所需train和test
$ python makeTxt.py 文件根目录

# 训练
$ python DistanceWeightedSampling.py
```



## 参考

- [DistanceWeightedSampling](https://github.com/suruoxi/DistanceWeightedSampling)

- [multigrain](https://github.com/facebookresearch/multigrain)



## 讨论

- dataloader的设计：假设一个batch_size=40，随机取8个类，那么每个类取5个样本，这样构成一个batch。继承了data的Sampler类，进行有目的的采样

- DistanceWeightedSampling：在一个batch内，根据两两之间的相似度作为采样的概率，来选择合适的negtive样本

- margin loss：跟传统的triplet loss不同是增加了一个beta，进行可训练的来调节样本之间的margin

  <img src = "https://st-gdx.dancf.com/gaodingx/39347986/design/mega/20190521-214244-61c8.jpg" width="100%"/>
  
  

