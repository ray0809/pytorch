from tqdm import tqdm
import numpy as np


def cal_map(labels, feats, topK):
    
    average_precision = 0.0
    for i in tqdm(range(feats.shape[0])):
        feat = feats[i]
        result = np.sum(np.square(feats - feat), axis=-1)
        # print('result',result.shape)
        #返回的第一个结果是自己本身，所以从1开始计数
        sort_indx = np.argsort(result)[1:topK+1] # 取top20的结果

        same_label_indx = (labels[sort_indx] == labels[i])

        if same_label_indx.sum() == 0:
            continue

        average_precision += ((np.cumsum(same_label_indx) / np.linspace(1, topK, topK)) * same_label_indx).sum() / same_label_indx.sum()

    mean_average_precision = average_precision / feats.shape[0]
    return mean_average_precision
