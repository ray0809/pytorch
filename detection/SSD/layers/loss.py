import torch
import torch.nn as nn
import torch.nn.functional as F
 
import numpy as np






class SSDLoss(nn.Module):
    def __init__(self):
        super(SSDLoss, self).__init__()
        self.focal_loss = focal_loss(size_average=True)

    def forward(self, 
                predict_bboxes, 
                predict_labels,
                gt_bboxes,
                gt_labels,
                neg_pos_ratio):
        '''
        输入：
        @predict_bboxes：(batch_size, num_prior, 4)    -> torch.tensor
        @predict_labels：(batch_size, num_prior, class_num)    -> torch.tensor
        @gt_bboxes：(batch_size, num_prior, 4)    -> torch.tensor
        @gt_labels：(batch_size, num_prior,)    -> torch.tensor
        '''
        
        
        # cross_entropy loss   sum(- y * log(softmax(pred)))
        # detach是为了断开反向求导，hard采样的过程不需要梯度回传
        class_num = predict_labels.size(-1)
        
        


        # 坐标回归只关心正样本，也就是具有标签类别的
        pos_mask = gt_labels > 0
        pos_predict_bboxes = predict_bboxes[pos_mask, :].reshape(-1, 4)
        pos_gt_bboxes = gt_bboxes[pos_mask, :].reshape(-1, 4)
        assert pos_predict_bboxes.shape == pos_gt_bboxes.shape
        loc_loss = F.smooth_l1_loss(pos_predict_bboxes, 
                                    pos_gt_bboxes, 
                                    reduction='sum')
        loc_num = pos_gt_bboxes.size(0)
        loc_loss  /= (loc_num + 1e-7)

        # 分类中，正负样本比为1:3
        # with torch.no_grad():
        log_softmax = -F.log_softmax(predict_labels, dim=2)[:,:,0].detach()
        mask = self.hard_neg_mining(log_softmax, gt_labels, neg_pos_ratio)
        
        mask_predict_labels = predict_labels[mask, :].reshape(-1, class_num)
        mask_gt_labels = gt_labels[mask].reshape(-1)
        conf_loss = F.cross_entropy(mask_predict_labels, 
                                    mask_gt_labels,
                                    reduction='sum')
        conf_loss /= (loc_num + 1e-7)
        # conf_loss = self.focal_loss(predict_labels, gt_labels)
        
        # conf_num = mask_gt_labels.size(0)
        # 两个损失都除的是正样本个数，等同于conf loss乘以一个(1+neg_pos_ratio)的系数了
        return loc_loss, conf_loss


        


    
    def hard_neg_mining(self, log_softmax, gt_labels, neg_pos_ratio):
        '''hard负样本采样，根据loss降序来得到topN的负样本，负:正=neg_pos_ratio
        输出：
        @mask：正样本和负样本的索引矩阵,需要计算loss的索引为1     -> torch.tensor.uint8(batch_size * num_prior)
        '''
        pos_mask = gt_labels > 0
        pos_num = pos_mask.sum(dim=1, keepdim=True)
        neg_num = pos_num * neg_pos_ratio

        
        # 正样本loss负无穷，这样降序，它会排在最后面
        log_softmax[pos_mask] = -np.inf
        # 背景loss降序排列，返回它们的索引
        _, sort_index = log_softmax.sort(dim=1, descending=True)

        # 再对sort_index排序单纯的只是为了得到一个矩阵，每行都是0～num_prior升序排列
        # 目的是为了得到一个mask，来得到sort_index的前neg_num个索引而已
        _, range_index = sort_index.sort(dim=1)
        neg_mask = range_index < neg_num
        return pos_mask | neg_mask




class focal_loss(nn.Module):
    '''
    # https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py
    '''
    def __init__(self, alpha=0.25, gamma=2, num_classes = 21, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss