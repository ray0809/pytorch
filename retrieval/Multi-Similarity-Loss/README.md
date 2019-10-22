## Sampling Matters in Deep Embedding Learning

multi-similarity loss

## 兼容

Python 3.6  
Torch 1.1.0  
Torchvison 0.3.0



## 更新

| Date       | Update     |
| ---------- | ---------- |
| 2019-10-22 | 第一次上传 |

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
| cifar10 | 0.9336 |





## 参考

- [research-ms-loss](https://github.com/MalongTech/research-ms-loss)

