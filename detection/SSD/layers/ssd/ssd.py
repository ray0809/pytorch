import torch
import torch.nn as nn




class SSD(nn.Module):
    '''ssd模型大框架，方便后续替换其他backbone
    输入：
    @base_net（nn.ModuleList）：基础模型，vgg、resnet等
    @base_layer_index（tuple）：需要接入reg和loc的层索引
    @extra_net（nn.ModuleList）：backbone后多接的数个层
    @conf_net，loc_net（nn.ModuleList）：分类层和回归层，需要接入到index和extra的每层后面，实现多尺度
    @cfg：配置参数文件，便于管理
    输出：
    @SSD：ssd模型（nn.Module）
    '''
    def __init__(self, base_net, 
                       base_layer_index,
                       extra_net,
                       conf_net,
                       loc_net,
                       # prior_bboxes,
                       cfg):
        super(SSD, self).__init__()
        
        self.base = base_net
        self.base_layer_index = base_layer_index
        # bn的作用是因为浅层的输出数值分布很分散，需要一个归一化
        self.bn = nn.BatchNorm2d(cfg.featmap_channels[0])
        self.extra_net = extra_net
        self.conf_net = conf_net
        self.loc_net = loc_net
        # self.prior_bboxes = prior_bboxes
        self.cfg = cfg

    
    def forward(self, x):
        confidences = []
        locations = []
        base_start_index = 0
        all_start_index = 0
        for i, index in enumerate(self.base_layer_index):
            for layer in self.base[base_start_index:index]:
                x = layer(x)
            if i == 0:
                y = self.bn(x)
            else:
                y = x

            conf, loc = self.compute_header(all_start_index, y)
            confidences.append(conf)
            locations.append(loc)
            
            base_start_index += index
            all_start_index += 1
        
        for layer in self.extra_net:
            x = layer(x)
            conf, loc = self.compute_header(all_start_index, x)
            confidences.append(conf)
            locations.append(loc)
            all_start_index += 1

        confidences = torch.cat(confidences, dim=1)
        locations = torch.cat(locations, dim=1)

        return confidences, locations#, self.prior_bboxes

        

    def compute_header(self, i, x):
        confidence = self.conf_net[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.cfg.class_num)


        location = self.loc_net[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location



    def load_pretrained_weights(self, pretrained_weights):
        
        model_dict = self.state_dict()  
        pretrained_dict = torch.load(pretrained_weights, map_location=torch.device('cpu'))
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)
        print(">>>load pretrained weights success, total {} layers.".format(len(pretrained_dict)))
        
    def init(self):
        def _xavier_init_(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        self.extra_net.apply(_xavier_init_)
        self.conf_net.apply(_xavier_init_)
        self.loc_net.apply(_xavier_init_)
        print(">>>init ssd success!")