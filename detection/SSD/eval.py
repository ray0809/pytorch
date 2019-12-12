import os
import cv2
import math
import torch
import random
import joblib
import numpy as np
from tqdm import tqdm

from config import config as cfg
from dataset import *
from layers.build import *
from layers.bbox_utils import *

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

torch.multiprocessing.set_sharing_strategy('file_system') 

def eval():

    VOC_CLASSES = [ 'background',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']

    trans = TestTransform()
    
    dataset = VOCDataset(root=cfg.root,
                            input_dim=cfg.input_dim,
                            tag='test',
                            years=cfg.years,
                            transform=trans,
                            vocannotation=VOCAnnotation(keep_difficult=False),
                            debug=False)

    dataloader = DataLoader(dataset=dataset,
                           batch_size=cfg.batch_size,
                           shuffle=False,
                           num_workers=cfg.num_workers,
                           collate_fn=collate_fn,
                           pin_memory=False,
                           drop_last=False)
 

    # 第一步，构建模型
    ssd_model = build_vgg_ssd(cfg)
    state_dict = torch.load(os.path.join(cfg.save_weights, 'best.pth'), 
                            map_location=lambda storage, loc: storage)
    ssd_model.load_state_dict(state_dict)
    ssd_model = ssd_model.to(cfg.device)
    ssd_model = ssd_model.eval()
    f_write = {}

    count = 0
    prior_bboxes = generator_prior_bboxes(cfg)
    for imgs, bboxes, labels, diffcults in tqdm(dataloader):

        with torch.no_grad():
            imgs = imgs.to(cfg.device)
            pred_conf, pred_loc = ssd_model(imgs)
            prior_bboxes = prior_bboxes.to(cfg.device)
            pred_loc = convert_offset_to_center(pred_loc, prior_bboxes)
            pred_loc = convert_center_to_corner(pred_loc)
            # print(pred_loc.shape, pred_conf.shape)
            batch = pred_loc.size(0)
            for b in range(batch):
                nms_bboxes, nms_probs, nms_labels = nms(pred_loc[b], 
                                                        pred_conf[b],
                                                        prob_threshold=0.0, 
                                                        iou_threshold=0.5,
                                                        topN=400)
            
                nms_bboxes_np = nms_bboxes.cpu().data.numpy()
                nms_labels_np = nms_labels.cpu().data.numpy()
                nms_probs_np = nms_probs.cpu().data.numpy()

                gt_bboxes_np = bboxes[b].cpu().data.numpy()
                gt_labels_np = labels[b].cpu().data.numpy()
                gt_diffcults_np = diffcults[b].cpu().data.numpy()

                f_write[count] = {'pred_bboxes':nms_bboxes_np,
                                  'pred_labels':nms_labels_np,
                                  'pred_prob':nms_probs_np,
                                  'gt_bboxes':gt_bboxes_np,
                                  'gt_labels':gt_labels_np,
                                  'gt_diffcults':gt_diffcults_np}

                count += 1
    
    joblib.dump(f_write, './weights/pred_test_dataset.joblib')


def cal_rec_pre(iou_threshold=0.5):
    data = joblib.load('./weights/pred_test_dataset.joblib')
    VOC_CLASSES = [ 'background',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']

    '''
    gts字典，key是类别索引，value字典（它的key是图片id，value是字典（它的key是该图片对应类别的bbox和tag标记））
    gts[label_i][img_id]['bboxes'], gts[label_i][img_id]['tag']
    
    preds字典，key是类别索引，value列表（装的元素是：[图片id, 类别概率, 类别bbox]，一个bbox对应一个元素）

    gts_count字典，key是类别索引，value是统计的每个类别的bboxes数量，用于后续的召回计算
    '''
    gts = {i:{} for i, _ in enumerate(VOC_CLASSES)}
    preds = {i:[] for i, _ in enumerate(VOC_CLASSES)}
    gts_count = {i:0 for i, _ in enumerate(VOC_CLASSES)}

    
    for img_id, value in data.items():
        pred_bboxes = value['pred_bboxes']
        pred_labels = value['pred_labels']
        pred_prob = value['pred_prob']
        gt_bboxes = value['gt_bboxes']
        gt_labels = value['gt_labels']
        set_labels = set(gt_labels.tolist())

        for j, pred_label in enumerate(pred_labels):
            preds[pred_label].append([img_id, pred_prob[j], pred_bboxes[j].reshape(1,4)])

        for j, label in enumerate(set_labels):
            gts[label][img_id] = {}
            index = gt_labels == label
            gts[label][img_id]['bboxes'] = gt_bboxes[index].reshape(-1, 4)
            gts[label][img_id]['tag'] = np.array([False] * index.sum())
            gts_count[label] += index.sum()

    '''
    step1:对于每个类别分开统计，首先将预测的preds[label_id]按照概率prob降序排列
    step2:对排序结果从头开始统计，如果能够跟gts中的bbox对应上（iou阈值符合）那么对应的gts下的tag置为true，
    也就是被挑选过，之后再也不会被选中了
    step3:记录与gts匹配中了多少个，每次迭代都要计算一次pre和rec，然后装到列表里
    '''
    MAP = 0.0
    for label, _ in enumerate(VOC_CLASSES):
        gt_count = gts_count[label]
        if gt_count > 0:
            iter_count = 0.0
            iter_pre, iter_rec = [], []
            confidences = sorted(preds[label], key=lambda x:x[1], reverse=True)
            for conf in confidences:
                img_id = conf[0]
                pred_bbox = conf[2]
                if img_id in gts[label]:
                    gt_bboxes = gts[label][img_id]['bboxes']
                    ious = jaccard_np(pred_bbox, gt_bboxes)[0]
                    max_idx = np.argmax(ious)
                    iou = ious[max_idx]
                    tag = gts[label][img_id]['tag'][max_idx]
                    if (iou > iou_threshold) and (tag == False):
                        gts[label][img_id]['tag'][max_idx] = True
                        iter_count += 1

                rec = iter_count / gt_count
                pre = iter_count / (len(iter_pre) + 1)
                iter_pre.append(pre)
                iter_rec.append(rec)
            iter_pre, iter_rec = np.array(iter_pre), np.array(iter_rec)
            cat_map = cal_map(iter_pre, iter_rec, use_07_metric=False)
            MAP += cat_map
            print('catgory: {}, map: {}'.format(VOC_CLASSES[label], cat_map))
    print('mean average precision: {}'.format(MAP / 20))


def cal_map(precision, recall, use_07_metric=False):
    '''计算mean average precision
    use_07_metric策略：
    将0～1划分为11个区间，在每个阈值下的rec右侧区域的最大pre作为该阈值的pre，然后记录这11个pre求平均
    use_10_metric策略：
    将统计到的recs划分区间（去重）右侧区域的最大pre作为该阈值的pre，以相邻rec的差作为底，max pre作为高求面积（近似积分）

    输入：
    @precision：从预测概率降序排列下的统计的pres,[pre0, pre1, pre2, ...]     ->   List
    @recall：从预测概率降序排列下的统计的recs,[rec0, rec1, rec2, ...]     ->   List
    
    输出：
    MAP：mean average precision， 标量    
    '''
    MAP = 0.0
    if use_07_metric:
        points = np.arange(0,1.1,0.1)

        ap = []
        for point in points:
            pres = precision[recall > point]
            if pres.size > 0:
                ap.append(np.max(pres))
            else:
                ap.append(0)
        
        MAP += sum(ap) / len(ap)
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        # 找每个样本点右区间的最大值
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        # 因为考虑的是样本点了，而不是固定大小区间，所以只找rec值不同的点（相同rec值的间隔点面积为0）
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        # 近似积分，间隔delta作为底，pre作为高，求面积
        MAP += np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
    return MAP




if __name__ == '__main__':
    os.system('clear')
    eval()
    cal_rec_pre()