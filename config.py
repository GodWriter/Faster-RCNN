import argparse
import yaml


def parse_args():
    """
    parsing and configuration
    :return: parse_args
    """
    desc = "TensorFlow implementation of fast-style-GAN"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--module', type=str, default='test',
                        help='Module to select: train, test, test_dataset, create_dataset, train_without_affine')
    parser.add_argument('--training', type=bool, default=False,
                        help='If the model is train, this argument should be true, else False')
    parser.add_argument('--GPU', type=str, default='0',
                        help='GPU used to train the model')
    parser.add_argument('--imageset', type=str, default='',
                        help='imageset splits')
    parser.add_argument('--config', type=str, default='config/config.yml',
                        help='Path of config file for testing')

    return parser.parse_args()


def read_conf_file(conf_file):
    class Args(object):
        def __init__(self, **entries):
            self.__dict__.update(entries)

    with open(conf_file) as f:
        args = Args(**yaml.load(f))
    return args
