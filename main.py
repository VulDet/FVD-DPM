import torch
import argparse
import time
from parser import Parser
from trainer import Trainer
import torch.distributed as dist
import yaml
from easydict import EasyDict as edict


def main(work_type_args):
    args = Parser().parse()
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config = edict(yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader))
    config.seed = args.seed
    config.dataset = args.dataset
    config.do_train = args.do_train

    if torch.distributed.get_rank() == 0:
        print(config)
    trainer = Trainer(config)

    if config.do_train == "train":
        trainer.train(ts)
    else:
        trainer.test(ts)
            

if __name__ == '__main__':
    dist.init_process_group("nccl", world_size=1)   # world_size is equal to gpu_number
    work_type_parser = argparse.ArgumentParser()
    x = work_type_parser.parse_known_args()
    main(work_type_parser.parse_known_args()[0])
