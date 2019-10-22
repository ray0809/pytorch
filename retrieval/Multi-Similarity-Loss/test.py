import os
import sys
import joblib
import shutil
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import DataLoader

from util import cal_map
from model import ResNet
from data_loader import TripletDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



TRANSFORMS = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])




def main(weight_path, joblib_path, text_path):
    model = ResNet(embedding_dim=128)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()

    testgen = TripletDataset(text_path, transforms=TRANSFORMS, return_path=True)
    testloader = DataLoader(dataset=testgen,
                            batch_size=100,
                            num_workers=10,
                            pin_memory=False)


    feats = []
    targets = []
    paths = []

    if not os.path.isfile(joblib_path):
        print('>>>开始提取test数据集的特征')
        with torch.no_grad():
            for (imgs, target, path) in tqdm(testloader):
                imgs = imgs.to(device)
                target = target.numpy().tolist()
                feat = model(imgs).detach().cpu().numpy().tolist()
                feats.extend(feat)
                targets.extend(target)
                paths.extend(path)

            joblib.dump([feats, targets, paths], joblib_path)
    else:
        print('>>>加载test数据集的特征')
        feats, targets, paths = joblib.load(joblib_path)


    feats, targets = np.array(feats), np.array(targets)
    
    print('>>>开始计算test数据集的mAP')
    mAP = cal_map(feats, targets, topK=10)
    print('>>>计算得到的mAP：{:.4f}'.format(mAP))

    nb = random.randint(0, feats.shape[0]-1)
    label = targets[nb]
    feat = feats[nb]
    sim = -np.dot(feats, feat)
    sort = np.argsort(sim)[1:10]

    save_path = './results/{}/'.format(nb)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        shutil.copy(paths[nb], os.path.join(save_path, 'label_{}_query_{}'.format(label, os.path.basename(paths[nb]))))

    for i, indx in enumerate(sort):
        p = paths[indx]
        shutil.copy(p, os.path.join(save_path, 'label_{}_index_{}_{}'.format(targets[indx], i, os.path.basename(p))))
    print('>>>检索结果已经写入results文件夹内') 
    



if __name__ == '__main__':
    weight_path, joblib_path, text_path= sys.argv[1], sys.argv[2], sys.argv[3]

    main(weight_path, joblib_path, text_path)