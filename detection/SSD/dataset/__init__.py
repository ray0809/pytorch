
import torch
from torch.utils.data import DataLoader
from .transformer import *
from .transformer1 import *
from .voc_dataset import *


def collate_fn(batch):
    tensor_imgs = []
    tensor_bboxes = []
    tensor_labels = []
    tensor_difficults = []
    for img, bboxes, labels, difficults in batch:
        # print('len bboxes: ', len(bboxes))
        if bboxes != []:
            
            tensor_imgs.append(torch.from_numpy(img.transpose(2,0,1)))
            # print(img.dtype, tensor_imgs[-1].dtype)
            tensor_bboxes.append(torch.FloatTensor(bboxes))
            tensor_labels.append(torch.LongTensor(labels))
            tensor_difficults.append(torch.LongTensor(difficults))
    tensor_imgs = torch.stack(tensor_imgs, dim=0)
    
    return tensor_imgs, tensor_bboxes, tensor_labels, tensor_difficults



# def create_dataloader(cfg, transform=None):
#     dataset = VOCDataset(pic_root=cfg.temp_pic_root,
#                          xml_root=cfg.temp_xml_root,
#                          transform=transform)

#     dataloader = DataLoader(dataset=dataset,
#                            batch_size=cfg.batch_size,
#                            shuffle=True,
#                            num_workers=cfg.num_workers,
#                            collate_fn=collate_fn,
#                            pin_memory=True,
#                            drop_last=False)


#     return dataloader
