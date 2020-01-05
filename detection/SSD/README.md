## 自己动手撸SSD

pytorch 0.4+，往上的版本应该都支持，我自己测试过1.1和1.3


搭建ssd的流程大概如下  
1.生成器（详情参考dataset文件夹）  
总要先从voc的原始格式数据中读取到我们需要的信息，xml格式下的文件，用ElementTree进行解析，放到节点定位到bndbox，这部分代码参考了[ssd-pytorch-voc0712](https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py)，读取到的图像增广参考的则是[pytorch-ssd-data_preprocessing](https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/ssd/data_preprocessing.py)，这两个模块没必要造轮子了，打算花更多的时间是在后面的流程中用自己的理解去实现，所以生成器这块基本上是照搬了他们两者。

2.初始模型相关搭建（参考layers文件夹）  
- basenet：vgg，主干网络，包括了全连接部分(用卷积来替代)，权重来源[pretrained-weight](https://github.com/amdegroot/ssd.pytorch#training-ssd)  
- normalize：原始论文是建立一个新层，只针对channel维度进行放缩，外加一个可学习权重进行加权，这里直接采用现有的BN层替代
- extra：在基础网络之后接了5个额外的模块(两个卷积两个relu)
- confs，locs：在每个extra模块之后都接入一个conf和一个loc，分别进行类别预测和坐标回归
- loss：主要主是有一个hard sample，conf进行交叉熵损失，loc进行smoothL1损失
>一、学写了torch.nn.ModuleList的使用，如果想灵活的从中间层跳出接另外的层，使用它更加合适。  
>二、为什么回归要用smoothL1：[请问 faster RCNN 和 SSD 中为什么用smooth L1 loss，和L2有什么区别？](https://www.zhihu.com/question/58200555)
>三、loss层的编写过程，如何从预测结果中拿到分类错误最大的topN的负样本，mask的获取。


3.bbox相关处理
- 先验框：不同的层感受野大小不一，所以预测的目标框大小也不同，决定框属性的有size和ratio两个，一个决定面积大小，一个决定长宽比，这方面的设计参考原论文。
- gt_bbox和gt_label：这两个也是重点，如何生成我们训练所需的bbox和标签
- offset：实际训练是回归proposal_box到gt_box的变换系数
- iou计算：target_box(num_target * 4)，prior_box(num_prior * 4)，通过一个矩阵计算得到(num_target * num_prior)

>一、gt_bbox和gt_label的生成的思想就是将先验框与每张图实际存在的bbox进行iou计算，符合一定阈值的才认为是正样本，同时赋予对应的标签，这里还有一个注意的是，每个真实的bbox与最接近的先验框是一定要对应上的，因为每个真实box必定是需要至少一个与之对应的，剩下的先验框赋予最接近的真实box，那么最终只需要按照iou来将低于阈值的置为label为0即可
>二、为什么不直接对坐标进行回归而是对offset，[直接回归目标的坐标是什么原理？](https://www.zhihu.com/question/304307091/answer/544905898?utm_source=com.tencent.tim&utm_medium=social&utm_oi=722819296425701376)，看的是最多赞的回答，大框和小框即使是同样的距离差，在小图框视觉上肯定会很大，大图框则对小偏差无感，回归变换系数，则能够同等对待。注意这里回归系数还会存在一个center_variance和size_variance，它们存在的目的是为了放大offset，让训练能更方便收敛。

4.后处理
- nms：non-maximum-suppression，非极大值抑制，ssd预测出来的框数量具有数千个，所以需要后处理去除掉冗余的框。
- mAP：mean average pricision，[目标检测中mAP的含义](https://www.zhihu.com/question/53405779)

>一、预测出来的数千个框(这里有个关键，对于所有的框，我们是按照预测的每个categroy对所有预测框都做一次nms，而不是对预测概率最大的那个categroy进行)。我们先按照预测的概率进行降序排列，然后取最大概率的框，剩余和它iou符合阈值的都剔除，然后接着剔除后剩下的概率最大的继续走下去
>二、目标检测的衡量指标是PR(precision,recall)曲线下的面积，准确率:你预测的结果中正确的占比=TP/(TP+FP)；召回率：你预测的正样本占所有正样本的比例=TP/(TP+FN)。

||P|N|
|-----|-----|-----|
|T|TP|TN|
|F|FP|FN|