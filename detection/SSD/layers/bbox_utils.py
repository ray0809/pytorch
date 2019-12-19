
import torch
import torch.nn.functional as F
import numpy as np
from math import *
from itertools import product


def generator_prior_bboxes(cfg, corner=True):
    '''生成先验框，不同层的featmap每个像素对应原图的大小不同，这个结果是固定的，
    所以只需要程序第一次运行得到即可，后续一直拿来用就好了
    以vgg为例：
    第一层：38*38
    第二层：19*19
    第三层：10*10
    第四层：5*5
    第五层：3*3
    第六层：1*1
    所以总共的先验框：4*38^2 + 6*19^2 + 6*10^2 + 6*5^2 + 4*3^2 + 4*1^1 = 8732

    输出：
    @prior_bboxes：先验框,已经将其还原回原始尺寸，而不是比例
    （shape：8732*4，坐标格式为：[center_x, center_y, w, h]）-> torch.tensor
    '''
    prior_bboxes = []
    for k, feat_size in enumerate(cfg.featmap_size):
        # receptivefield = cfg.input_dim / cfg.steps[k]
        ratio = cfg.ratios[k]
        down_size, up_size = cfg.sizes[k], cfg.sizes[k+1]
        sqrt_size = sqrt(down_size * up_size)
        # 全排列，生成featmap的每个坐标
        for j, i in product(range(feat_size), repeat=2):
            center_x = (i + 0.5) / feat_size
            center_y = (j + 0.5) / feat_size

            prior_bboxes.append([center_x, center_y, down_size, down_size])
            prior_bboxes.append([center_x, center_y, sqrt_size, sqrt_size])


            for rat in ratio:
                prior_bboxes.append([center_x, 
                                     center_y, 
                                     down_size / sqrt(rat), 
                                     down_size * sqrt(rat)])

                prior_bboxes.append([center_x, 
                                     center_y, 
                                     down_size * sqrt(rat), 
                                     down_size / sqrt(rat)])

    prior_bboxes = torch.tensor(prior_bboxes)
    prior_bboxes.clamp_(min=0.0, max=1.0)
    prior_bboxes *= cfg.input_dim
    if corner:
        prior_bboxes = convert_center_to_corner(prior_bboxes)
    return prior_bboxes


def jaccard(gt_bbox, prior_bboxes):
    '''计算目标框与先验框的iou 
    计算方式：(A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B))
    输入：
    @gt_bbox：单张图的目标框数   -> torch.tensor(num_target, 4)
    @priro_bboxes：先验框的数量  -> torch.tensor(num_prior,4)

    输出：
    @overlap：每个目标针对所有先验框的iou   -> torch.tensor(num_target, num_prior)
    '''
    num_target = gt_bbox.size(0)
    num_prior = prior_bboxes.size(0)
    gt_bbox = gt_bbox.unsqueeze(1).expand(num_target, num_prior, 4)
    prior_bboxes = prior_bboxes.unsqueeze(0).expand(num_target, num_prior, 4)

    min_xy = torch.max(gt_bbox[..., :2], prior_bboxes[..., :2])
    max_xy = torch.min(gt_bbox[..., 2:], prior_bboxes[..., 2:])

    overlap = torch.clamp(max_xy - min_xy, min=0.0)
    overlap = overlap[..., 0] * overlap[..., 1]

    area0 = torch.clamp(gt_bbox[..., 2:] - gt_bbox[..., :2], min=0.0)
    area0 = area0[..., 0] * area0[..., 1]

    area1 = torch.clamp(prior_bboxes[..., 2:] - prior_bboxes[..., :2], min=0.0)
    area1 = area1[..., 0] * area1[..., 1]

    return overlap / (area0 + area1 - overlap + 1e-8)






def assert_prior_bboxes(gt_bbox, gt_label, prior_bboxes, iou_threshold=0.5):
    '''根据每个输入图像的真实目标框来选择先验框，通过之间的iou阈值来选择正负样本
    输入：
    @gt_bbox：[coord1, coord2, ...], 其中coordx表示四个点坐标   -> List
    @gt_label：[12, 10, ...], 同上，对应的是每个框所属的类别      -> List 
    @priro_bboxes：先验框，这里坐标装载的是四个点的坐标([x1, y1, x2, y2])，而不是中心    -> torch.tensor(8732, 4)
    @iou_threshold：先验框与目标框的iou阈值，满足才将对应先验框置为正         -> float
    输出：
    @bboxes：每个先验框对应的真实目标的框([x1, y1, x2, y2])，需要根据labels来确定哪些是有用于loss的   -> torch.tensor(num_prioir, 4)
    @labels：每个先验框的标签，为0的话说明该框为背景     -> torch.tensor(num_prioir,)
    '''
    

    # 如果是gt在前，那么返回的矩阵：num_target * num_prior
    ious = jaccard(gt_bbox, prior_bboxes)

    # tensor自带的min、max方法默认返回的value，index
    prior_per_target_iou, prior_per_target_index = ious.max(1)  # (num_target,)
    target_per_prior_iou, target_per_prior_index = ious.max(0)  # (num_prior,)

    for target_index, prior_index in enumerate(prior_per_target_index):
        target_per_prior_index[prior_index] = target_index

    # 统计到的与每个target占比最大的prior，这些先验框是必定需要的，所以对应的index提前赋值2
    # 那么这些框必定会挑选出来，2 > 1 >= iou_threshold
    target_per_prior_iou.index_fill_(0, prior_per_target_index, 2.0)

    # 最终，每个先验框都有一个目标框对应，只是说是否会被loss计算，看label的值便可
    labels = gt_label[target_per_prior_index]
    labels[target_per_prior_iou < iou_threshold] = 0
    bboxes = gt_bbox[target_per_prior_index]

    return bboxes, labels


def generator_grouth(gt_bboxes, gt_labels, prior_bboxes, iou_threshold=0.5):
    '''这是针对batch_size
    根据每个输入图像的真实目标框来选择先验框，通过之间的iou阈值来选择正负样本
    输入：
    @gt_bbox：[coord1, coord2, ...], 其中coordx表示四个点坐标   -> List
    @gt_label：[12, 10, ...], 同上，对应的是每个框所属的类别      -> List 
    @priro_bboxes：先验框，这里坐标装载的是四个点的坐标([x1, y1, x2, y2])，而不是中心    -> torch.tensor(8732, 4)
    @iou_threshold：先验框与目标框的iou阈值，满足才将对应先验框置为正         -> float
    输出：
    @bboxes：每个先验框对应的真实目标的框([x1, y1, x2, y2])，需要根据labels来确定哪些是有用于loss的   -> torch.tensor(batch_size, num_prioir, 4)
    @labels：每个先验框的标签，为0的话说明该框为背景     -> torch.tensor(batch_size, num_prioir)
    '''
    new_bboxes, new_labels = [], []
    batch_size = len(gt_bboxes)
    for i in range(batch_size):
        bboxes, labels = assert_prior_bboxes(gt_bboxes[i], 
                                             gt_labels[i], 
                                             prior_bboxes,
                                             iou_threshold)

        new_bboxes.append(bboxes.unsqueeze(0))
        new_labels.append(labels.unsqueeze(0))


    return torch.cat(new_bboxes, dim=0), torch.cat(new_labels, dim=0)



def convert_cetner_to_offset(gbboxes, 
                             pbboxes, 
                             center_variance=0.1,
                             size_varivance=0.2,
                             corner=True):
    '''中心宽高坐标 -> 训练过程需要的标签(offset)
    参考原论文上列出的计算方式(g:grouth bbox; p:prior bbox)：
    n_cx = (g_cx - p_cx) / p_w
    n_cy = (g_cy - p_cy) / p_h
    n_w = log(g_w / p_w)
    n_h = log(g_h / p_h)

    输入：
    @gbboxes：grouth bbox，[center_x, center_y, w, h]    -> torch.tensor(batchs_size, num_prior, 4)
    @pbboxes：prior bbox，[center_x, center_y, w, h]    -> torch.tensor(num_prior, 4)
    @center_variance：对center xy进行缩放    -> float
    @size_varivance：对wh进行缩放     -> float
    @corner：如果为true则说明输入的tensor是corner的格式    -> bool
    参考：https://zhuanlan.zhihu.com/p/39399799
    除以variance是对预测box和真实box的误差进行放大，从而增加loss，增加梯度，加快收敛速度
    @corner：输入是否是基于四个点坐标的形式，true的话需要将它们在内部转为center的形式   -> bool
    输出：
    @bboxes：[n_cx, n_cy, n_w, n_h]     -> torch.tensor(num_prior, 4)
    '''
    if corner:
        gbboxes = convert_corner_to_center(gbboxes)
        pbboxes = convert_corner_to_center(pbboxes)
    n_cxcy = (gbboxes[..., :2] - pbboxes[..., :2]) / (center_variance * pbboxes[..., 2:])
    n_wh = torch.log(gbboxes[..., 2:] / pbboxes[..., 2:]) / size_varivance
    
    return torch.cat([n_cxcy, n_wh], -1) 


def convert_offset_to_center(goffset, 
                             pbboxes, 
                             center_variance=0.1, 
                             size_varivance=0.2,
                             corner=True):
    '''训练过程需要的标签(offset)  ->  中心宽高坐标
    具体的计算公式根据convert_cetner_to_offset列出的进行逆推回去
    输入：
    @goffset：[g_cx, g_cy, g_w. g_w]     -> torch.tensor(batch_size, num_prior, 4)
    @pbboxes：先验框[x1, y1, x2, y2]     ->  torch.tensor(num_proir, 4)
    输出：
    @偏移转为正常坐标：[n_cx, n_cy, n_w. n_w]     -> torch.tensor(num_prior, 4)
    
    '''
    if corner:
        pbboxes = convert_corner_to_center(pbboxes)
    
    n_cxcy = goffset[..., :2] * center_variance * pbboxes[..., 2:] + pbboxes[..., :2]
    n_wh = torch.exp(goffset[..., 2:] * size_varivance) * pbboxes[..., 2:]

    return torch.cat([n_cxcy, n_wh], -1) 



def convert_corner_to_center(bboxes):
    '''左上右下坐标 -> 中心宽高坐标
    输入：
    @bboxes：[x1, y1, x2, y2]    -> torch.tensor
    输出：
    @bboxes：[center_x, center_y, w, h]    -> torch.tensor
    '''
    new_bboxes = torch.cat([(bboxes[..., 2:] + bboxes[..., :2])/2,
                            (bboxes[..., 2:] - bboxes[..., :2])],
                            dim=-1)
    return new_bboxes


def convert_center_to_corner(bboxes):
    '''中心宽高坐标 -> 左上右下坐标
    输入：
    @bboxes：[center_x, center_y, w, h]   -> torch.tensor
    输出：
    @bboxes：[x1, y1, x2, y2]    -> torch.tensor
    '''
    new_bboxes = torch.cat([bboxes[..., :2] - bboxes[..., 2:]/2,
                            bboxes[..., :2] + bboxes[..., 2:]/2],
                            dim=-1)
    return new_bboxes
    


def jaccard_np(bbox1, bbox2):
    '''基于numpy的iou计算
    输入：
    @bbox1，bbox2：两个需要计算iou的矩形框（n1*4）和（n2*4）   ->  numpy.ndarray
    输出：
    @iou：n1*n2，不同框之间的iou
    '''
    batch1 = bbox1.shape[0]
    batch2 = bbox2.shape[0]
    bbox1 = np.tile(np.expand_dims(bbox1, 1), (1, batch2, 1))
    bbox2 = np.tile(np.expand_dims(bbox2, 0), (batch1, 1, 1))
    min_xy = np.maximum(bbox1[..., :2], bbox2[..., :2])
    max_xy = np.minimum(bbox1[..., 2:], bbox2[..., 2:])

    rec = np.clip(max_xy - min_xy, a_min=0.0, a_max=np.inf)

    overlap = rec[..., 0] * rec[..., 1]

    rec1 = bbox1[..., 2:] - bbox1[..., :2]
    rec2 = bbox2[..., 2:] - bbox2[..., :2]

    area1 = rec1[..., 0] * rec1[..., 1]
    area2 = rec2[..., 0] * rec2[..., 1]

    iou = overlap / (area1 + area2 - overlap + 1e-8)

    return iou    # (batch1, batch2)



def nms(pred_loc, pred_conf, prob_threshold=0.0, iou_threshold=0.5, topN=200):
    '''非极大值抑制
    输入：
    @pred_loc：模型的目标框回归结果，前提是已经offset->corner了    -> torch.tensor(num_prior, 4)
    @pred_conf：模型的目标框预测结果，不需要额外的前置      -> torch.tensor(num_prior, num_class)
    @prob：挑选预测概率大于prob的预测结果来进行非极大值抑制
    @topN：挑选预测概率降序后的topN个框来进行非极大值抑制
    输出：
    @final_bboxes：最终预测的目标框[x1,y1,x2,y2]     -> torch.tensor(num_filted, 4)
    @final_labels：最终预测的类别        ->   torch.tensor(num_filted,)
    @final_probs：最终预测的概率        ->   torch.tensor(num_filted,)
    '''
    pred_softmax = F.softmax(pred_conf, dim=-1)   # softmax操作（num_prior, num_class）
    full_pred_prob, full_pred_label = pred_softmax.max(-1)   # （num_prior,）

    # 这里得到的数据是将预测为背景的都滤除了
    filter_mask = full_pred_label > 0   # 剔除掉预测为背景的,剩余数量（num_filted,）
    pred_prob = full_pred_prob[filter_mask]
    pred_label = full_pred_label[filter_mask]
    pred_bboxes = pred_loc[filter_mask]   # （num_filted, 4）


    # 预测概率降序排列得到排列的索引，为了减小计算量，可以只取topN的数据
    _, idx = pred_prob.sort(dim=-1, descending=True)
    # idx = idx[:topN]    # 预测概率降序的topN的索引


    keep = []   # 用来装载nms之后的可用的索引
    while(idx.numel() > 0):
        keep.append(idx[0])   # 概率最大的可用
        bbox = pred_bboxes[idx[0]].reshape(1, 4)  # 最大概率的框坐标

        idx = idx[1:]    # 剔除最大的
        other_bboxes = pred_bboxes[idx].reshape(-1, 4)

        iou = jaccard(bbox, other_bboxes)[0]   # 大小(1, len(idx[1:]))
        idx = idx[iou < iou_threshold]   # 符合iou阈值的都是要剔除的，剩余的才能走接下来的循环

    keep = torch.LongTensor(keep).to(pred_loc.device)

    nms_bboxes = torch.index_select(pred_bboxes, 0, keep)
    nms_probs = torch.index_select(pred_prob, 0, keep)
    nms_labels = torch.index_select(pred_label, 0, keep)

    
    return nms_bboxes, nms_probs, nms_labels




def nms1(pred_loc, pred_conf, prob_threshold=0.0, iou_threshold=0.5, topN=200):
    '''非极大值抑制
    输入：
    @pred_loc：模型的目标框回归结果，前提是已经offset->corner了    -> torch.tensor(num_prior, 4)
    @pred_conf：模型的目标框预测结果，不需要额外的前置      -> torch.tensor(num_prior, num_class)
    @prob：挑选预测概率大于prob的预测结果来进行非极大值抑制
    @topN：挑选预测概率降序后的topN个框来进行非极大值抑制
    输出：
    @final_bboxes：最终预测的目标框[x1,y1,x2,y2]     -> torch.tensor(num_filted, 4)
    @final_labels：最终预测的类别        ->   torch.tensor(num_filted,)
    @final_probs：最终预测的概率        ->   torch.tensor(num_filted,)
    '''
    pred_loc = pred_loc.to(torch.device('cpu'))
    pred_conf = pred_conf.to(torch.device('cpu'))

    pred_softmax = F.softmax(pred_conf, dim=-1)   # softmax操作（num_prior, num_class）

    class_num = pred_conf.size(-1)
    nms_bboxes, nms_probs, nms_labels = [], [], []

    for i in range(1, class_num):
        probs = pred_softmax[:, i]  # （num_prior,）
        mask = probs > prob_threshold

        # 这里得到的数据是将概率低于阈值的过滤
        pred_prob = probs[mask].reshape(-1)
        pred_bboxes = pred_loc[mask].reshape(-1, 4)   # （num_filted, 4）


        # 预测概率降序排列得到排列的索引，为了减小计算量，可以只取topN的数据
        _, idx = pred_prob.sort(dim=-1, descending=True)
        idx = idx[:topN]    # 预测概率降序的topN的索引


        keep = []   # 用来装载nms之后的可用的索引
        while(idx.numel() > 0):
            keep.append(idx[0])   # 概率最大的可用
            bbox = pred_bboxes[idx[0]].reshape(1, 4)  # 最大概率的框坐标

            idx = idx[1:]    # 剔除最大的
            other_bboxes = pred_bboxes[idx].reshape(-1, 4)

            iou = jaccard(bbox, other_bboxes)[0]   # 大小(1, len(idx[1:]))
            idx = idx[iou < iou_threshold]   # 符合iou阈值的都是要剔除的，剩余的才能走接下来的循环
        
        if keep != []:
            temp_labels = torch.LongTensor([i] * len(keep)).to(pred_loc.device)
            keep = torch.LongTensor(keep).to(pred_loc.device)
            

            nms_bboxes.append(torch.index_select(pred_bboxes, 0, keep))
            nms_probs.append(torch.index_select(pred_prob, 0, keep))
            nms_labels.append(temp_labels)
    
    if nms_bboxes != []:
        nms_bboxes = torch.cat(nms_bboxes, 0)
        nms_probs = torch.cat(nms_probs, 0)
        nms_labels = torch.cat(nms_labels, 0)
    else:
        nms_bboxes = pred_loc.new()
        nms_probs = pred_loc.new()
        nms_labels = pred_loc.new()
    
    return nms_bboxes, nms_probs, nms_labels

