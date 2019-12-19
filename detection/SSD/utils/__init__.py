import torch
from collections import OrderedDict


def load_state_dict(weight_path):
    '''保存模型有时候忘了是多gpu还是单gpu
    （它们的区别就在于state_dict的key有没有多出一个module）
    所以统一加载的时候用cpu的方式
    '''
    old_state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:] # remove `module.`
        new_state_dict[name] = v

    return new_state_dict