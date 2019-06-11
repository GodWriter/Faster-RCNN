import os
import numpy as np
import xml.etree.ElementTree as ET

from symnet.logger import logger
from .imdb import IMDB


class PascalVOC(IMDB):
    classes = ['__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, image_set, root_path, devkit_path):
        super(PascalVOC, self).__init__('voc_' + image_set, root_path)

        year, image_set = image_set.split('_')
        self._config = {'comp_id': 'comp4',
                        'use_diff': False,
                        'min_size': 2}
        self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self._image_index_file = os.path.join(devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
        self._image_file_tmpl = os.path.join(devkit_path, 'VOC' + year, 'JPEGImages', '{}.jpg')
        self._image_anno_tmpl = os.path.join(devkit_path, 'VOC' + year, 'Annotations', '{}.xml')

        # print info
        logger.info('class_to_ind: %s' % self._class_to_ind)
        logger.info('class_to_ind: %s' % self._image_index_file)

        # results
        result_folder = os.path.join(devkit_path, 'results', 'VOC' + year, 'Main')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        self._result_file_tmpl = os.path.join(result_folder, 'comp4_det_' + image_set + '_{}.txt')

        # get roidb
        # self._roidb = self._get_cached('roidb', self._load_gt_roidb)
        self._roidb = self._load_gt_roidb()
        logger.info('%s num_images %d' % (self.name, self.num_images))

    def _load_gt_roidb(self):
        # get the image index for each image
        image_index = self._load_image_index()
        # through the index ,load the related xml
        gt_roidb = [self._load_annotation(index) for index in image_index]
        return gt_roidb

    def _load_image_index(self):
        with open(self._image_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def _load_annotation(self, index):
        # get the source height, width, objs
        height, width, orig_objs = self._parse_voc_anno(self._image_anno_tmpl.format(index))

        if not self._config['use_diff']:
            non_diff_objs = [obj for obj in orig_objs if obj['difficult'] == 0]
            objs = non_diff_objs
        else:
            objs = orig_objs
        # get the number of objects
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = obj['bbox'][0] - 1
            y1 = obj['bbox'][1] - 1
            x2 = obj['bbox'][2] - 1
            y2 = obj['bbox'][3] - 1
            cls = self._class_to_ind[obj['name'].lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        roi_rec = {'index': index,
                   'objs': objs,
                   'image': self._image_file_tmpl.format(index),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'flipped': 0}

        return roi_rec

    @staticmethod
    def _parse_voc_anno(filename):
        tree = ET.parse(filename)
        height = int(tree.find('size').find('height').text)
        width = int(tree.find('size').find('width').text)
        objects = []

        for obj in tree.findall('object'):
            obj_dict = dict()
            obj_dict['name'] = obj.find('name').text
            obj_dict['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                                int(float(bbox.find('ymin').text)),
                                int(float(bbox.find('xmax').text)),
                                int(float(bbox.find('ymax').text))]
            objects.append(obj_dict)

        return height, width, objects

