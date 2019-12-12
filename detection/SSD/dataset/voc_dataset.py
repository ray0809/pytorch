import os
import cv2
import math
import random
import os.path as opth
import numpy as np
from glob import glob
from torch.utils import data
import xml.etree.ElementTree as ET




class VOCAnnotation():
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult
        # 左上角和右下角
        self.pts = ['xmin', 'ymin', 'xmax', 'ymax']
        self.VOC_CLASSES = [ 'background',
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor']


    def __call__(self, xml_file):
        root = ET.parse(xml_file).getroot()
        size = root.find('size')
        filename = root.find('filename').text
        coordinates, labels, difficults = [], [], []
        # height, width = int(size.find('height')), int(size.find('width'))
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            difficults.append(difficult)
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            
            bndbox = []
            for pt in self.pts:
                bndbox.append(float(bbox.find(pt).text) - 1)

            # 标签的数字编码
            label = self.VOC_CLASSES.index(name)
            labels.append(label)
            coordinates.append(bndbox)

        return filename, coordinates, labels, difficults


class VOCDataset(data.Dataset):
    '''voc数据生成器，包含解析xml得到矩形框，以及读图
    @img:图像
    @bboxes: n*4([x1, y1, x2, y2])
    '''
    def __init__(self, 
                 root,
                 input_dim, 
                 tag='trainval', 
                 years=['VOC2007', 'VOC2012'], 
                 vocannotation=VOCAnnotation(),
                 transform=None, 
                 debug=True):
        self.root = root
        self.years = years
        if debug:
            self.xml_files, self.img_files = self._tempfindinfo(root, years)
        else:
            self.xml_files, self.img_files = self._findinfo(root, years, tag)
        self.transform = transform
        self.vocannotation = vocannotation
        self.input_dim = input_dim
        self.tag = tag


    def _findinfo(self, root, years, tag):
        '''获取所有的xml和图像的路径
        '''
        
        
        xml_files = []
        img_files = []
        for year in years:
            txt_path = opth.join(root, year, 'ImageSets', 'Main', '{}.txt'.format(tag))
            if opth.isfile(txt_path):
                for line in open(txt_path, 'r'):
                    line = line.strip()
                    img_files.append(opth.join(root, year, 'JPEGImages', '{}.jpg'.format(line)))
                    xml_files.append(opth.join(root, year, 'Annotations', '{}.xml'.format(line)))

        return xml_files, img_files

    def _tempfindinfo(self, root, years):
        '''单纯的是用来测试本地代码的正确性
        '''
        xml_files = []
        img_files = []
        for year in years:
            img_files += glob(opth.join(root, year, 'JPEGImages', '*.jpg'))
            xml_files += glob(opth.join(root, year, 'Annotations', '*.xml'))

        return xml_files, img_files


    def __getitem__(self, index):
        xml_file = self.xml_files[index]
        img_name, bboxes, labels, difficults = self.vocannotation(xml_file)
        # img_path = os.path.join(self.pic_root, img_name)
        img = cv2.imread(self.img_files[index], 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(bboxes) == 0:
            return img, [], []
        bboxes = np.array(bboxes)
        labels = np.array(labels)
        if self.transform is not None:
            # annotations = {'image': img, 
            #                'bboxes': bboxes,
            #                'labels': labels}
            # annotations = self.transform(annotations)
            # img, bboxes, labels =  annotations['image'], annotations['bboxes'], annotations['labels']
            # 增广出来的坐标是percent百分比的，所以如果要输出绝对坐标，需要乘图像宽高
            img, bboxes, labels = self.transform(img, bboxes, labels)
        img = img.astype('float32')
        return img, bboxes.tolist(), labels.tolist(), difficults

    def __len__(self):
        return len(self.xml_files)



if __name__ == '__main__':

    # test for VOCAnnotation
    # xml_file = './test_voc_data/2012_004251.xml'
    # annotation_parse = VOCAnnotation(keep_difficult=False)
    # objects = annotation_parse(xml_file)
    # print(objects)


    pass

