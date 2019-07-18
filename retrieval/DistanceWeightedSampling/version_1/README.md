## Sampling Matters in Deep Embedding Learning

针对triplet三元组的采样策略，核心就在于一个batch中的negtive sample的选择

## 兼容

Python 3.6  
Torch 1.1.0  
Torchvison 0.3.0



## 更新

| Date       | Update     |
| ---------- | ---------- |
| 2019-07-18 | 第一次上传 |

## 数据

如果想训练你自己的数据，那么请参考txt文件夹下的格式



## 训练

```
python train.py --train_txt 你的训练txt路径 --test_txt 你的测试txt路径 --gpu_ids 0
```

## 测试

```
python test.py 权重路径, 特征保存路径（第一次会先提取特征再保存，后续不会再做提取操作）测试txt路径
```

## 实验

| Dataset      | mAP(topK=10)     |
| ---------- | ---------- |
| cifar10 | 0.9232 |


## 讨论
- 采样和loss的代码沿用了参考第三个的代码，其他几个都有训练不收敛的情况，如果你没遇到类似问题，烦请告知下，我请教请教
- dataloader的设计：假设一个batch_size=40，随机取8个类，那么每个类取5个样本，这样构成一个batch。继承了data的Sampler类，进行有目的的采样
- 该方法的核心在于negtive sample的选择策略
- margin loss：跟传统的triplet loss不同的是增加了一个beta，进行可训练的来调节样本之间的margin

![dir_network](https://st-gdx.dancf.com/gaodingx/39347986/design/mega/20190521-214244-61c8.jpg)
  

关于beta，看原论文初始化一个10000维的向量，但实际计算loss的时候也只是一个batch_size内考虑，也就是0～batch_size-1的范围而已，所以我理解的是10000是比较随意的，只要大于等于batch size的大小即可，所以我的实验中是设置成了跟批量大小一致，通过这个可学习的软间隔来缓解刚性的margin边界。
起初我使用的代码是参考2的，他把beta设置成的是一个可训练的标量，尝试了sgb和adam优化器，模型不同层使用不同的学习率，实验过程发现不收敛
后来看到了参考3的代码，进行了尝试，效果显著



## 参考

- [DistanceWeightedSampling](https://github.com/suruoxi/DistanceWeightedSampling)

- [multigrain](https://github.com/facebookresearch/multigrain)

- [DeepEmbedding](https://github.com/hudengjunai/DeepEmbeding/blob/master/models/sample_dml.py)
