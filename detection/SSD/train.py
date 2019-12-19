import os
import cv2
import math
import random
import torchsnooper
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR

from utils import *
from dataset import *
from layers.build import *
from layers.loss import *
from layers.bbox_utils import *
from config import config as cfg

torch.backends.cudnn.benchmark = True

def train(cfg):
    # 第一步，构建模型
    ssd_model = build_vgg_ssd(cfg)
    ssd_model.init()
    pretrained_dict = torch.load(cfg.vgg_pretrained_weights, map_location=torch.device('cpu'))
    ssd_model.base.load_state_dict(pretrained_dict)
    ssd_model.to(cfg.device)
    
    ssd_model = torch.nn.DataParallel(ssd_model)
    ssd_model = ssd_model.train()

    # 第二步，构建数据生成器
    aug = TrainAugmentation()
    # aug = Transformer()
    dataset = VOCDataset(root=cfg.root,
                            input_dim=cfg.input_dim,
                            years=cfg.years,
                            transform=aug,
                            debug=False)

    dataloader = DataLoader(dataset=dataset,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           num_workers=cfg.num_workers,
                           collate_fn=collate_fn,
                           pin_memory=True,
                           drop_last=False)

    # 第三步，定义损失
    criterion = SSDLoss()

    # 第四步，定义优化器
    optimizer = optim.SGD(ssd_model.parameters(), 
                          lr=cfg.lr, 
                          momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    scheduler = MultiStepLR(optimizer, cfg.milestones, cfg.gamma, -1)

    
    prior_bboxes = generator_prior_bboxes(cfg)
    batch_iterator = iter(dataloader)
    tqdm_iters = tqdm(range(0, cfg.max_iter))

    min_loss = np.inf
    average_loss = 0.0
    average_loc = 0.0
    average_conf = 0.0

    # with torchsnooper.snoop():
    for its in tqdm_iters:
        scheduler.step()
        
        try:
            imgs, bboxes, labels, _ = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader)
            imgs, bboxes, labels, _ = next(batch_iterator)


        imgs = imgs.to(cfg.device)

        pred_conf, pred_loc = ssd_model(imgs)
        # dataloader出来的bbox是针对img上的绝对坐标
        gt_bboxes, gt_labels = generator_grouth(bboxes, 
                                                labels, 
                                                prior_bboxes,
                                                cfg.iou_threshold)

        gt_offset = convert_cetner_to_offset(gt_bboxes, prior_bboxes, corner=True)
        gt_offset = gt_offset.to(cfg.device)
        gt_labels = gt_labels.to(cfg.device)

        loc_loss, conf_loss = criterion(pred_loc,
                                        pred_conf,
                                        gt_offset,
                                        gt_labels,
                                        cfg.neg_pos_ratio)
        
        optimizer.zero_grad()
        loss = loc_loss + conf_loss
        
        average_loss += loss.item()
        average_loc += loc_loss.item()
        average_conf += conf_loss.item()

        loss.backward()
        optimizer.step()
        tqdm_iters.set_description("iter {}".format(its+1))
        tqdm_iters.set_postfix(loc=loc_loss.item(),
                                conf=conf_loss.item(),
                                total=loss.item())

        if (its + 1) % cfg.check_iter == 0:
            average_loss /= cfg.check_iter
            average_loc /= cfg.check_iter
            average_conf /= cfg.check_iter
            print('\n>>>iter: {}, \
                    average loss: {:.4f}, \
                    loc loss: {:.4f}, \
                    conf loss: {:.4f}'.format(its+1, 
                                        average_loss,
                                        average_loc,
                                        average_conf))
            if average_loss < min_loss:
                min_loss = average_loss
                # torch.save(ssd_model.module.state_dict(), os.path.join(cfg.save_weights, 'best.pth'))
                torch.save(ssd_model.state_dict(), cfg.save_weights)
            
            average_loss = 0.0
            average_loc = 0.0
            average_conf = 0.0

if __name__ == '__main__':
    os.system('clear')
    train(cfg)