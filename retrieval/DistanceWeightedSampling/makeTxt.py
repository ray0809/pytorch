#coding=utf-8
import os
import sys
import random
from tqdm import tqdm

if __name__ == '__main__':
    #数据根目录
    ROOT = sys.argv[1]

    IDS = []
    SUB = []
    
    for root, dirs, files in os.walk(ROOT):
        label = root.split('/')[-1]
        IDS.append(label)
        SUB.append(root)



    imgs = []
    labels = []
    if not os.path.isdir('./txt'):
        os.makedirs('./txt')
    with open('./txt/train.txt', 'w') as w:
        with open('./txt/test.txt', 'w') as m:
            for label, (ids, subpath) in tqdm(enumerate(zip(IDS, SUB))):
                files = os.listdir(subpath)
                random.shuffle(files)
                train = files[:int(len(files) * 0.9)]
                test = files[int(len(files) * 0.9):]

                for f in train:
                    path = os.path.join(subpath, f)
                    if (os.path.isfile(path)) and ('jpg' in path):
                        w.write(path + ' ' + str(label) + '\n')
                for f in test:
                    path = os.path.join(subpath, f)
                    if (os.path.isfile(path)) and ('jpg' in path):
                        m.write(path + ' ' + str(label) + '\n')

    # 随机打乱数据
    os.system("shuf \"./txt/train.txt\" -o \"./txt/train.txt\" ")
    os.system("shuf \"./txt/test.txt\" -o \"./txt/test.txt\" ")