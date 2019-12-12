import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import math
import random
import joblib
import numpy as np
from tqdm import tqdm
import tensorwatch as tw
from torchsummary import summary
from tensorboardX import SummaryWriter

from config import config as cfg
from dataset import *
from layers.build import *
from layers.bbox_utils import *

def test_augmentation():
    # 验证数据增强

    trans = TrainAugmentation()
    vocdataset = VOCDataset(root=cfg.temp_root,
                            input_dim=cfg.input_dim,
                            years=cfg.years,
                            transform=trans,
                            debug=True)

    img, bboxes, labels, _ = vocdataset.__getitem__(1)
    img *= 255
    img = img.astype('uint8')
    print(bboxes, labels)
    n = len(bboxes)
    for i in range(n):
        xmin, ymin, xmax, ymax = [math.floor(j) for j in bboxes[i][:4]]
        cv2.rectangle(img, 
                      (xmin, ymin), 
                      (xmax, ymax),
                      color=(255,0,0),thickness=3)
    cv2.imshow('test', img.astype('uint8'))
    cv2.waitKey(0)


def test_dataloader():
    # 验证生成器
    trans = TrainAugmentation()

    dataset = VOCDataset(root=cfg.temp_root,
                            input_dim=cfg.input_dim,
                            years=cfg.years,
                            transform=trans,
                            bebug=True)

    dataloader = DataLoader(dataset=dataset,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           num_workers=cfg.num_workers,
                           collate_fn=collate_fn,
                           pin_memory=True,
                           drop_last=False)
    dataloader = iter(dataloader)
    imgs, bboxes, labels = next(dataloader)
    print(imgs.size())
    print(bboxes)


def test_build_vgg():
    # 验证vgg主干构建生成
    backbone = create_vgg_backbone(cfg)
    extra = add_extra()
    model = nn.Sequential(*(backbone + extra))
    # g = tw.draw_model(model.eval(), [1,3,300,300])
    # g.save('vgg_network.pdf') 
    print(tw.model_stats(model.eval(), [1,3,300,300]))
    # print(model)


def test_build_ssd():
    # 验证ssd模型构建成功
    ssd = build_vgg_ssd(cfg)
    ssd = ssd.eval()
    
    inputs = torch.randn(1,3,300,300)

    conf, loc, _ = ssd(inputs)
    print('conf size: {}, loc size: {}'.format(conf.size(), loc.size()))
    g = tw.draw_model(ssd.eval(), [1,3,300,300])
    g.save('ssd_network.png') 
    # print(tw.model_stats(ssd, [1,3,300,300]))

def load_pretrained():
    ssd_model = build_vgg_ssd(cfg)
    print('init: ', ssd_model.state_dict()['base.33.weight'][0,0,:,:])
    pretrained_dict = torch.load(cfg.vgg_pretrained_weights, map_location=torch.device('cpu'))
    print('pretrained: ', pretrained_dict['33.weight'][0,0,:,:])
    ssd_model.base.load_state_dict(pretrained_dict)
    print('load pretrained: ', ssd_model.state_dict()['base.33.weight'][0,0,:,:])





def test_generate_prior():
    
    # 验证先验框正确生成
    prior_bbox = generator_prior_bboxes(cfg, corner=False)
    print("absolute coord(center): ", prior_bbox[2000:2004])

    prior_bbox = convert_center_to_corner(prior_bbox)
    print("absolute coord(corner): ", prior_bbox[2000:2004])
    print('prior require gard: ', prior_bbox.requires_grad)

    prior_bbox = prior_bbox.numpy()

    bg = np.zeros((300,300,3), dtype='uint8')
    for i in range(2000, 2004):
        xmin, ymin, xmax, ymax = [math.floor(j) for j in prior_bbox[i][:4]]
        cv2.rectangle(bg, 
                      (xmin, ymin), 
                      (xmax, ymax),
                      color=(255,0,0))
    cv2.imshow('test', bg)
    cv2.waitKey(0)
        



def test_generate_grouth():
    # 验证基于先验框和iou筛选得到剩余的grouth框
    # 验证基于先验框和iou筛选得到剩余的grouth框
    trans = TestTransform()
    
    dataset = VOCDataset(root=cfg.temp_root,
                        input_dim=cfg.input_dim,
                        years=cfg.years,
                        transform=trans)

    dataloader = DataLoader(dataset=dataset,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           num_workers=cfg.num_workers,
                           collate_fn=collate_fn,
                           pin_memory=False,
                           drop_last=False)
    dataloader = iter(dataloader)
    imgs, bboxes, labels = next(dataloader)
    
    idx = 3

    prior_bbox = generator_prior_bboxes(cfg, corner=False)
    prior_bbox = convert_center_to_corner(prior_bbox)
    # gt_bboxes, gt_labels = assert_prior_bboxes(bboxes[0], labels[0], prior_bbox, cfg.iou_threshold)
    gt_bboxes, gt_labels = generator_grouth(bboxes, labels, prior_bbox, cfg.iou_threshold)

    print("gt_labels sum", (gt_labels[idx] > 0).sum())
    print(gt_bboxes[idx].shape, gt_labels[idx].shape)
    gt_bboxes = prior_bbox[gt_labels[idx] > 0]
    gt_bboxes_np = gt_bboxes.numpy()

    first_img = imgs[idx].permute(1, 2, 0).numpy().copy()
    first_img = first_img.astype('uint8')
    first_bboxes = bboxes[idx].numpy()





    n = gt_bboxes_np.shape[0]
    bb = []
    for i in range(n):
        xmin, ymin, xmax, ymax = [math.floor(j) for j in gt_bboxes_np[i][:4]]
        temp = [xmin, ymin, xmax, ymax]
        if temp not in bb:
            bb.append(temp)
    
    # temp = np.zeros((300,300,3), dtype='uint8')

    for b in bb:
        # print(b)
        xmin, ymin, xmax, ymax = b
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        cv2.rectangle(first_img, (xmin, ymin), (xmax, ymax), (b,g,r))


    bb = np.array(bb)
    o = bb.shape[0]
    m = first_bboxes.shape[0]
    for j in range(m):
        temp = first_bboxes[j].astype('int32')
        print('temp', temp)
        
        xmin, ymin, xmax, ymax = temp
        cv2.rectangle(first_img, (xmin, ymin), (xmax, ymax), (0,0,255), 3)

        for k in range(o):
            print('{}: iou:{}'.format(bb[k], jaccard_np(temp.reshape(1,4), bb[k].reshape(1,4))))

    cv2.imshow('test', first_img)
    cv2.waitKey(0)


def test_convert_corner_center():
    # 验证坐标转换，四点<->中心
    prior_bbox = generator_prior_bboxes(cfg, corner=False)
    print('ori center: ', prior_bbox[0])

    prior_bbox_corner = convert_center_to_corner(prior_bbox)
    print('corner: ', prior_bbox_corner[0])

    prior_bbox_center = convert_corner_to_center(prior_bbox_corner)
    print('center: ', prior_bbox_center[0])


def test_generate_offset():
    # 验证生成训练所需的offset

    trans = TestTransform()
    
    dataset = VOCDataset(root=cfg.temp_root,
                        input_dim=cfg.input_dim,
                        years=cfg.years,
                        transform=trans)

    dataloader = DataLoader(dataset=dataset,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           num_workers=cfg.num_workers,
                           collate_fn=collate_fn,
                           pin_memory=False,
                           drop_last=False)
    dataloader = iter(dataloader)
    imgs, bboxes, labels, _ = next(dataloader)

    prior_bbox_center = generator_prior_bboxes(cfg, corner=False)

    prior_bbox_corner = convert_center_to_corner(prior_bbox_center)
    # gt_bboxes, gt_labels = assert_prior_bboxes(bboxes[0], labels[0], prior_bbox_corner, cfg.iou_threshold)
    gt_bboxes, gt_labels = generator_grouth(bboxes, labels, prior_bbox_corner, cfg.iou_threshold)
    print("gt_labels sum: ", (gt_labels[0] > 0).sum().item())
    print("gt_bboxes: ", gt_bboxes[0][:4])

    gt_offset = convert_cetner_to_offset(gt_bboxes, prior_bbox_corner, corner=True)

    print("gt_offset: ", gt_offset[0][:4])

    gt_bboxes_back = convert_offset_to_center(gt_offset, prior_bbox_corner, corner=True)
    gt_bboxes_back_corner = convert_center_to_corner(gt_bboxes_back)
    print("gt_bboxes_back_corner: ", gt_bboxes_back_corner[0][:4])


def test_nms():
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
                            debug=False)

    dataloader = DataLoader(dataset=dataset,
                           batch_size=cfg.batch_size,
                           shuffle=True,
                           num_workers=cfg.num_workers,
                           collate_fn=collate_fn,
                           pin_memory=False,
                           drop_last=False)
    dataloader = iter(dataloader)

    imgs, bboxes, labels, _ = next(dataloader)

    # 第一步，构建模型
    ssd_model = build_vgg_ssd(cfg)
    ssd_model.init()
    state_dict = torch.load(os.path.join(cfg.save_weights, 'best.pth'), 
                            map_location=lambda storage, loc: storage)
    ssd_model.load_state_dict(state_dict)
    # ssd_model.to(cfg.device_cpu)

    ssd_model = ssd_model.eval()
    f_write = {}

    # for imgs, bboxes, labels in tqdm(dataloader):
    # imgs = imgs.to(cfg.device_cpu)

    with torch.no_grad():
        pred_conf, pred_loc, prior_bboxes = ssd_model(imgs)
        pred_loc = convert_offset_to_center(pred_loc, prior_bboxes)
        pred_loc = convert_center_to_corner(pred_loc)
        # prior_bboxes = prior_bboxes.to(cfg.device_cpu)
        # print(pred_loc.shape, pred_conf.shape)
        nms_bboxes, nms_probs, nms_labels = nms(pred_loc[0], 
                                                pred_conf[0],
                                                prob_threshold=0.0, 
                                                iou_threshold=0.5,
                                                topN=200)
    
        nms_bboxes_np = nms_bboxes.data.numpy()
        nms_labels_np = nms_labels.data.numpy()
        nms_probs_np = nms_probs.data.numpy()
        num = nms_bboxes_np.shape[0]

        gt_bboxes_np = bboxes[0].data.numpy()
        gt_labels_np = labels[0].data.numpy()
        gt_num = gt_bboxes_np.shape[0]

        for j in range(gt_num):
            bbox = gt_bboxes_np[j]
            label = gt_labels_np[j]
            print(">>>")
            print("    label: ", VOC_CLASSES[label])
            print("    bbox: ", bbox)

        for i in range(num):
            bbox = nms_bboxes_np[i]
            label = nms_labels_np[i]
            prob = nms_probs_np[i]
            print(">>>")
            print("    predict category: {}, pro: {:.4f}".format(VOC_CLASSES[label], prob))
            print("    predict bbox: {}".format(bbox))
            # xmin, ymin, xmax, ymax = [math.floor(b) for b in bbox]




def test_predict():
    VOC_CLASSES = [ 'background',
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']

    img = cv2.imread('./dummyImgs/2.jpg', 1)
    img = cv2.resize(img, (300,300))
    img_float = img.astype('float32')
    img_float /= 255.0
    img_tensor = torch.from_numpy(img_float).permute(2,0,1).unsqueeze(0)

    # 第一步，构建模型
    ssd_model = build_vgg_ssd(cfg)
    ssd_model.init()
    state_dict = torch.load(os.path.join(cfg.save_weights, 'best.pth'), 
                            map_location=lambda storage, loc: storage)
    ssd_model.load_state_dict(state_dict)
    ssd_model = ssd_model.eval()
    prior_bboxes = generator_prior_bboxes(cfg)
    with torch.no_grad():
        pred_conf, pred_loc = ssd_model(img_tensor)
        pred_loc = convert_offset_to_center(pred_loc, prior_bboxes)
        pred_loc = convert_center_to_corner(pred_loc)

        # print(pred_loc.shape, pred_conf.shape)
        nms_bboxes, nms_probs, nms_labels = nms(pred_loc[0], 
                                                pred_conf[0],
                                                prob_threshold=0.0, 
                                                iou_threshold=0.4,
                                                topN=200)
        
        
        nms_bboxes_np = nms_bboxes.data.numpy()
        nms_labels_np = nms_labels.data.numpy()
        nms_probs_np = nms_probs.data.numpy()
        num = nms_bboxes_np.shape[0]
        for i in range(num):
            
            bbox = nms_bboxes_np[i]
            label = nms_labels_np[i]
            prob = nms_probs_np[i]
            print(">>>")
            print("    predict category: {}, pro: {:.4f}".format(VOC_CLASSES[label], prob))
            print("    predict bbox: {}".format(bbox))
            xmin, ymin, xmax, ymax = [math.floor(b) for b in bbox]
            
            cv2.rectangle(img, 
                            (xmin, ymin), 
                            (xmax, ymax),
                            color=(255,0,0),thickness=2)
    cv2.imshow('test', img)
    cv2.waitKey(0)


if __name__ == '__main__':

    os.system("clear")

    load_pretrained()




    