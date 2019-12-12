import torch
import torch.nn as nn

def create_vgg(vgg_backbone, cfg):
    
    vgg_backbone_layer_index = (cfg.first_layer_index, len(vgg_backbone))
    
    base_net = nn.ModuleList(vgg_backbone)
    extra_net = nn.ModuleList(add_extra())
    loc_net, conf_net = add_reg_loc(cfg)


    return vgg_backbone_layer_index, base_net, extra_net, conf_net, loc_net





def add_extra():
    layers = [
        nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                        nn.ReLU()),
        nn.Sequential(nn.Conv2d(512, 128, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                        nn.ReLU()),
        nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, kernel_size=3),
                        nn.ReLU()),
        nn.Sequential(nn.Conv2d(256, 128, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, kernel_size=3),
                        nn.ReLU())
    ]
    return layers


def add_reg_loc(cfg):
    location_layers = nn.ModuleList([
        nn.Conv2d(cfg.featmap_channels[0], cfg.bboxes_num[0] * 4, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[1], cfg.bboxes_num[1] * 4, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[2], cfg.bboxes_num[2] * 4, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[3], cfg.bboxes_num[3] * 4, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[4], cfg.bboxes_num[4] * 4, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[5], cfg.bboxes_num[5] * 4, kernel_size=3, padding=1)
    ])

    confidence_layers = nn.ModuleList([
        nn.Conv2d(cfg.featmap_channels[0], cfg.bboxes_num[0] * cfg.class_num, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[1], cfg.bboxes_num[1] * cfg.class_num, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[2], cfg.bboxes_num[2] * cfg.class_num, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[3], cfg.bboxes_num[3] * cfg.class_num, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[4], cfg.bboxes_num[4] * cfg.class_num, kernel_size=3, padding=1),
        nn.Conv2d(cfg.featmap_channels[5], cfg.bboxes_num[5] * cfg.class_num, kernel_size=3, padding=1)
    ])

    return location_layers, confidence_layers 



