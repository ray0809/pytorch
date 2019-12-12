

from .backbone import *
from .ssd import *
from .loss import *
from .bbox_utils import *


def build_vgg_ssd(cfg):
    vgg_backbone = create_vgg_backbone(cfg, batch_norm=False)
    vgg_backbone_layer_index, base_net, extra_net, conf_net, loc_net = create_vgg(vgg_backbone, cfg)

    # prior_bboxes = generator_prior_bboxes(cfg)
    VGG_SSD = SSD(base_net, 
                    vgg_backbone_layer_index,
                    extra_net, 
                    conf_net, 
                    loc_net,
                    # prior_bboxes,
                    cfg)

    return VGG_SSD