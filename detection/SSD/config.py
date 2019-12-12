import torch


class config:
    class_num = 21
    root = '/home/hanbing/hanbing_data/datasets/detection/VOCdevkit'
    years = ['VOC2007', 'VOC2012']
    vgg_pretrained_weights = './weights/vgg16_reducedfc.pth'
    save_weights = './weights/'

    # 自己模拟的一些测试数据，用于验证数据增广，生成器等
    temp_root = './dataset/test_voc_data/'


    


    # vgg layers config
    input_dim = 300
    iou_threshold = 0.5
    # 一种放大操作，对offset进行放大，确保loss不会太小
    center_variance = 0.1  
    size_variance = 0.2
    

    # vgg16
    vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                  512, 512, 512]

    featmap_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # vgg-backbone的浅层、最后一层以及额外添加的4层, index都是从1开始数的
    first_layer_index = 23
    featmap_channels = [512, 1024, 512, 256, 256, 256]

    

    # 不同层每个像素预测的框的宽高比
    ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # 不同层每个像素预测的框的面积占比（相对于input_dim）
    # 0.1是浅层的一个特殊比例，其他的是由 0.2～1.05划分成5个区间
    sizes = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
    # 不同层的每个像素预测的框数量
    # 不同的size和ratio组合而成,以ratios[0]为例：
    # 实际上它含有[1,2,1/2],对应第一个比例区间[0.1,0.2]
    # 它有 2+3-1=4个框，其他的依次类推
    bboxes_num = [4, 6, 6, 6, 4, 4]


    # 训练参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    batch_size = 64
    num_workers = 8
    epoches = 200
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    neg_pos_ratio = 3
    # 选择迭代到对应步数进行lr衰减：lr *= gamma^step
    milestones = [120, 160]
    gamma = 0.1

