import os
import sys
import numpy as np
import tensorflow as tf

from symimdb.pascal_voc import PascalVOC
from symnet.logger import logger


class Dataset(object):
    def __init__(self,
                 args):
        self.args = args
        self.tfrecordnames = []

        for tfrecord in os.listdir(self.args.tfrecord_file):
            self.tfrecordnames.append(os.path.join(self.args.tfrecord_file, tfrecord))

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
        file_created = 0
        file_saved = 0

        roidb = self.get_voc()
        print("roidb: ", roidb[1])
        while file_created < self.args.tfrecord_num:
            tf_filename = '%s/train_%03d.tfrecord' % (self.args.tfrecord_file,
                                                      file_saved)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                file_created_per_record = 0
                while file_created < self.args.tfrecord_num and file_created_per_record < self.args.samples_per_file:
                    sys.stdout.write('\r>> Converting image %d/%d' % (file_created, self.args.tfrecord_num))
                    sys.stdout.flush()
                    add_to_tfrecord(roidb[file_created], tfrecord_writer)
                    file_created += 1
                    file_created_per_record += 1
                file_saved += 1

        print('\nFinished converting to the tfrecord.')

    def load_dataset(self, data_list):
        dataset = tf.data.TFRecordDataset(data_list)
        new_dataset = dataset.map(parse_function)
        shuffle_dataset = new_dataset.shuffle(buffer_size=10000)
        batch_dataset = shuffle_dataset.batch(self.args.batch_size)
        epoch_dataset = batch_dataset.repeat(self.args.num_epochs)

        iterator = epoch_dataset.make_initializable_iterator()

        return iterator

    def test_dataset(self):
        data_list = tf.placeholder(tf.string, shape=[None])
        iterator = self.load_dataset(data_list)
        example = iterator.get_next()

        count = 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer, feed_dict={data_list: self.tfrecordnames})
            while True:
                try:
                    exmaple_ = sess.run(example)
                except tf.errors.OutOfRangeError:
                    print("End of dataSet")
                    break
                else:
                    print('No.%d' % count)
                    print('bbox: ', exmaple_['image/bbox'])
                    print('gt_classes: ', exmaple_['image/bbox/label'])
                    print('flipped: ', exmaple_['image/flipped'])
                    print('image/format: ', exmaple_['image/format'])
                    print('image.shape: ', exmaple_['image/encoded'].shape)
                count += 1


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def add_to_tfrecord(roi, tfrecord_writer):
    shape = [roi['height'], roi['width'], 3]
    image_data = tf.gfile.GFile(roi['image'], 'rb').read()

    # xmin, ymin, xmax, ymax = [], [], [], []
    # for box in roi['boxes']:
    #     [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], box)]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/bbox': bytes_feature(roi['boxes'].tostring()),
        'image/bbox_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=roi['boxes'].shape)),
        'image/bbox/label': tf.train.Feature(int64_list=tf.train.Int64List(value=roi['gt_classes'])),
        'image/flipped': int64_feature(roi['flipped']),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)
    }))

    tfrecord_writer.write(example.SerializeToString())


def parse_function(example_proto):
    keys_to_features = {
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/bbox': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/bbox_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
        'image/bbox/label': tf.FixedLenFeature(shape=(1,), dtype=tf.int64),
        'image/flipped': tf.FixedLenFeature([1], tf.int64),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    parsed_example = tf.parse_single_example(example_proto, keys_to_features)

    parsed_example['image/bbox'] = tf.decode_raw(parsed_example['image/bbox'], tf.uint16)
    parsed_example['image/bbox'] = tf.reshape(parsed_example['image/bbox'], parsed_example['image/bbox_shape'])

    parsed_example['image/encoded'] = tf.image.decode_jpeg(parsed_example['image/encoded'])

    return parsed_example
