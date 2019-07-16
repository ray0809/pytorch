
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
from torch.utils.data import Dataset


'''
Triplet翻译过来是三元组的意思，分别表示选中样本，一个正样本，一个负样本
训练：
同类的距离越近越好，异类的距离越远越好
'''
class TripletDataset(Dataset):
    def __init__(self, txt, tag='train', transforms=None):
        self.data,self.labels = self._parsing(txt)
        self.tag = tag
        self.transforms = transforms
        self.label_set = set(self.labels)
        self.label2index = { i:list(np.where(np.array(self.labels)==i)[0]) for i in self.label_set}
        if tag != 'train':
            '''
            依次遍历整个数据集，第i个数据，作为它的正pair，随机从所有该类别内选取一个，负样本随机选取一个异类，再随机选取其中一个样本。
            '''
            triplets = [[i, 
                        random.choice(self.label2index[self.labels[i]]), 
                        random.choice(self.label2index[random.choice(list(self.label_set - set([self.labels[i]])))])
                        ] 
                        for i in tqdm(range(len(self.data)))]
            

            
            self.test_triplet = triplets
        
    def __getitem__(self, index):
        if self.tag == 'train':
            img1 = self._read(self.data[index])
            label = self.labels[index]
            
            postive_index = index
            while postive_index == index:
                postive_index = random.choice(self.label2index[label])
            img2 = self._read(self.data[postive_index])
            
            negtive_label = random.choice(list(self.label_set - set([label])))
            negtive_index = random.choice(self.label2index[negtive_label])
            img3 = self._read(self.data[negtive_index])
            
        else:
            sample = self.test_triplet[index]
            # print(sample)
            img1 = self._read(self.data[sample[0]])
            img2 = self._read(self.data[sample[1]])
            img3 = self._read(self.data[sample[2]])
            # print(img1.shape, img2.shape, img3.shape)
        
        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
            img3 = self.transforms(img3)

            
        return img1, img2, img3
    

    def _read(self, path):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if w != h:
            MAX = max(w, h)
            img = ImageOps.expand(img, border=(0, 0, MAX - w, MAX - h), fill=0)

        return img

    
    
    def _parsing(self, txt):
        paths = []
        labels = []
        with open(txt, 'r') as f:
            f = f.readlines()
            for i in f:
                path, label = i.strip().split(' ')
                paths.append(path)
                labels.append(int(label))
        return paths, labels
    
    def __len__(self):
        if self.tag == 'train':
            return len(self.data)
        else:
            return len(self.test_triplet)





