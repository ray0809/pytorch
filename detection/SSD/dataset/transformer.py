from albumentations import *


# 该库对目标检测的增强，可以将label放置在框的四个点的尾部，构成长度为5的向量
def get_aug(augs, format='pascal_voc', min_area=0., min_visibility=0.):
    return Compose(augs, bbox_params=BboxParams(format=format, 
                                               min_area=min_area, 
                                               label_fields=['labels'],
                                               min_visibility=min_visibility
                                               ))


# https://github.com/albumentations-team/albumentations/blob/master/notebooks/example_bboxes.ipynb
class Transformer():
    def __init__(self):
        self.aug = get_aug(augs=[HorizontalFlip(p=1),
                                 ChannelShuffle(p=1),
                                 HueSaturationValue(p=1),
                                 RandomBrightnessContrast(p=1),
                                 RandomResizedCrop(height=300,width=300),
                                 CLAHE(p=0.5)
                                 ],
                            min_visibility=0.2)


    def __call__(self, annotations):
        augmented = self.aug(**annotations)
        return augmented


