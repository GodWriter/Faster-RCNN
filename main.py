import os

from config import parse_args, read_conf_file
from dataloader import Dataset


args_cmd = parse_args()
args_yml = read_conf_file(args_cmd.config)

if __name__ == '__main__':
    module = args_cmd.module

    os.environ['CUDA_VISIBLE_DEVICES'] = args_cmd.GPU

    if module == 'create_dataset':
        dataset = Dataset(args_yml)
        dataset.create_dataset()
    elif module == 'test_dataset':
        dataset = Dataset(args_yml)
        dataset.test_dataset()
    else:
        pass
