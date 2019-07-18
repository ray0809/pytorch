import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler



class BalancedBatchSampler(Sampler):
    def __init__(self, txt, batch_size, batch_k):
        assert (batch_size % batch_k == 0 ) and (batch_size > 0)
        self.dataset = {}
        self.balanced_max = 0
        self.batch_size = batch_size
        self.batch_k = batch_k

        
        # Save all the indices for all the classes
        count = 0
        with open(txt, 'r') as f:
            for i, j in enumerate(f):
                j = j.strip().split()
                if j[1] not in self.dataset:
                    self.dataset[j[1]] = []
                self.dataset[j[1]].append(i)
                count += 1
        
        num_samples = [len(value) for value in self.dataset.values()]
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)
        self.length = count // batch_size

        assert self.min_samples >= self.batch_k
    
        self.keys = list(self.dataset.keys())
        #self.currentkey = 0

    def __iter__(self):
        batch = []
        for i in range(self.length):
            classes = random.sample(range(len(self.keys)), k=int(self.batch_size/self.batch_k))
            for cls in classes:
                cls_idxs = self.dataset[self.keys[cls]]
                batch.extend(random.sample(cls_idxs, k=self.batch_k))
            
            yield batch
            batch = []

    def __len__(self):
        return self.length




class TripletDataset(Dataset):
    def __init__(self, txt, transforms=None, return_path=False):
        self.transforms = transforms
        self.return_path = return_path
        self.img_paths = []
        self.labels = []
        with open(txt, 'r') as f:
            for i, j in enumerate(f):
                j = j.strip().split()
                self.img_paths.append(j[0])
                self.labels.append(int(j[1]))
                
    def __getitem__(self, idx):
        
        if self.transforms:
            img = self.transforms(self._read(self.img_paths[idx]))
        else:
            img = self._read(self.img_paths[idx])
            
        label = self.labels[idx]
        
        if self.return_path:
            return img, label, self.img_paths[idx]
        else:
            return img, label
            
    
    def _read(self, path):
        img = Image.open(path).convert('RGB')
        w, h = img.size
        if w != h:
            MAX = max(w, h)
            img = ImageOps.expand(img, border=(0, 0, MAX - w, MAX - h), fill=0)

        return img
 
    
    def __len__(self):
        return len(self.img_paths)




class TripletDataset1(Dataset):
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