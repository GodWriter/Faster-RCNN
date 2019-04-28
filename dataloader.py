import numpy as np

from symimdb.pascal_voc import PascalVOC
from symnet.logger import logger


class Dataset(object):
    def __init__(self,
                 args):
        self.args = args

    def get_voc(self):
        if not self.args.imageset:
            self.args.imageset = '2012_trainval'
        self.args.rcnn_num_classes = len(PascalVOC.classes)

        isets = self.args.imageset.split('+')
        roidb = []
        for iset in isets:
            # absorb source gt
            imdb = PascalVOC(iset, 'data', 'data/VOCdevkit')
            # delete useless dt
            imdb.filter_roidb()
            # data enlarge
            imdb.append_flipped_images()
            roidb.extend(imdb.roidb)
        return roidb

    def create_dataset(self):
        roidb = self.get_voc()
        pass

    def test_dataset(self):
        roidb = self.get_voc()

        logger.info('len(roidb): %d' % len(roidb))
        print("An example: ", roidb[1])


