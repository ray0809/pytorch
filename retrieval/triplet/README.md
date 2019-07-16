## Sampling Matters in Deep Embedding Learning

基于triplet三元组，根据图像之间的相似概率进行采样选择positive和negtive样本  

## 兼容

Python 3.6  
Torch 1.1.0  
Torchvison 0.3.0
tqdm（进度条显示）
joblib(保存特征)



## 更新

| Date       | Update     |
| ---------- | ---------- |
| 2019-07-16 | 第一次上传 |

## 

## 数据

如果想训练你自己的数据，请参考txt文件夹内的格式




## 训练

```
# 训练
# 训练过程的权重会自动保存到weight文件夹内
$ python train.py

```
## 训练

```
# 测试
# 抽取特征的文件路径：如果第一次跑，会先进行特征抽取，然后再保存
# 测试数据集的txt路径：本次实验cifar10的话就直接制定txt下的test.txt路径就好
python test.py 权重路径 抽取特征的文件路径 测试数据集的txt路径
```

## 讨论
10000条数据的top10的mAP：0.8057

