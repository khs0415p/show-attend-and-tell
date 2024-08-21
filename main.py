
import os
import sys
import warnings
from sconf import Config
from argparse import ArgumentParser

import torch.distributed
import torch.multiprocessing.spawn
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from trainer import Trainer
from utils import setup_env


def load_config(config_path):
    config = Config(config_path)
    return config


def main(args):    
    if (args.continuous and not args.checkpoint) or (args.mode == "test" and not args.checkpoint):
        raise ValueError("Not Found - checkpoint path")
    
    if args.mode == 'train':
        config = load_config(args.config)
    else:
        config = load_config(os.path.join(args.checkpoint, 'config.yaml'))

    config.checkpoint = args.checkpoint
    config.continuous = args.continuous
    config.mode = args.mode

    setup_env()

    if len(config.device) <= 1 or config.device in ['cpu', 'cuda']:
        single_train(args, config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.device))
        ngpus_per_node = len(config.device)
        torch.multiprocessing.spawn(multi_train, nprocs=ngpus_per_node, args=(ngpus_per_node, config, args))


def single_train(args, config):
    if config.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:0") if config.device == "cuda" else torch.device(f"cuda:{config.device[0]}")

    trainer = Trainer(
        config,
        device,
        )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()


def multi_train(rank, ngpus_per_node, config, args):
    
    torch.distributed.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=rank)

    torch.cuda.set_device(rank)
    torch.distributed.barrier()

    trainer = Trainer(
        config,
        rank,
    )

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        raise NotImplementedError


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", '-c', type=str, help="The configuration file name.")
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--continuous', '-con', action="store_true", help="Continuous training from checkpoint")
    parser.add_argument("--checkpoint", '-ck', type=str, help="The checkpoint directory.")
    args = parser.parse_args()

    main(args)