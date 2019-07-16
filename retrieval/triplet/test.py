import os
import sys
import joblib
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps

from util import cal_map
from model import load_resnet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main(weight_path, joblib_path, text_path):
    model = load_resnet(num_classes=128)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    


    if os.path.isfile(joblib_path):
        print('>>>加载预先提取的特征数据')
        paths, labels, feats = joblib.load(joblib_path)
    
    else:
        print('>>>第一次运行，开始生成test数据集的特征向量')
        paths = []
        labels = []
        feats = []


        model.eval()
        with torch.no_grad():
            with open(text_path, 'r') as f:
                for i in tqdm(f.readlines()):
                    path, label = i.strip().split()
                    img = Image.open(path).convert('RGB').resize((224,224))
                    img = torch.from_numpy(np.expand_dims(np.array(img).astype('float32') / 255, 0).transpose(0,3,1,2)).to(device)
                    feat = model(img).data.cpu().numpy()[0]
                    paths.append(path)
                    labels.append(int(label))
                    feats.append(feat)

        labels = np.array(labels)
        feats = np.array(feats)
        joblib.dump([paths, labels, feats], joblib_path)


    print('>>>开始计算test数据集的mAP')
    mAP = cal_map(labels, feats, topK=10)
    print('>>>计算得到的mAP：{:.4f}'.format(mAP))


if __name__ == '__main__':
    weight_path, joblib_path, text_path= sys.argv[1], sys.argv[2], sys.argv[3]

    main(weight_path, joblib_path, text_path)