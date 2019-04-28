import numpy as np

from symimdb.pascal_voc import PascalVOC


class Dataset(object):
    def __init__(self,
                 args):
        self.args = args

    def get_voc(self):
        if not self.args.imageset:
            self.args.imageset = '2007_trainval'
        self.args.rcnn_num_classes = len(PascalVOC.classes)

        isets = self.args.imageset.split('+')
        roidb = []
        for iset in isets:
            imdb = PascalVOC(iset, 'data', 'data/VOCdevkit')
            imdb.filter_roidb()
            imdb.append_flipped_images()
            roidb.extend(imdb.roidb)
        return roidb
