from tqdm import tqdm
import numpy as np


def cal_map(feats, labels, topK):
    
    average_precision = 0.0
    results = np.argsort(-np.dot(feats, feats.T), axis=1)
    for i in range(feats.shape[0]):
        #返回的第一个结果是自己本身，所以从1开始计数
        sort_indx = results[i][1:topK+1] # 取topk0的结果
        same_label_indx = (labels[sort_indx] == labels[i])

        if same_label_indx.sum() == 0:
            continue

        average_precision += ((np.cumsum(same_label_indx) / np.linspace(1, topK, topK)) * same_label_indx).sum() / same_label_indx.sum()

    mean_average_precision = average_precision / feats.shape[0]
    return mean_average_precision