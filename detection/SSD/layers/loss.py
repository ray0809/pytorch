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
        with torch.no_grad():
            log_softmax = -F.log_softmax(predict_labels, dim=2)[:,:,0]
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

