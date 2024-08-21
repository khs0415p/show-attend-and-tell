import os
import torch
import random
import pandas as pd

from typing import List, Dict
from tokenizer import Tokenizer
from utils import  seed_worker
from utils.data_utils import CustomDataset
from torch.utils.data import DataLoader, random_split, distributed, RandomSampler, SequentialSampler


IGNORE_ID = -100


def collate_fn_warpper(padding_id):
    def collate_fn_inner(batch):
        return collate_fn(batch, padding_id)
    return collate_fn_inner


def collate_fn(batch: List[Dict[str, torch.Tensor]], padding_value: int = 0) -> Dict[str, torch.Tensor]:    
    outputs = {key : [instance[key] for instance in batch] for key in ['image', 'caption']}

    # Dynamic padding
    captions = torch.nn.utils.rnn.pad_sequence(
        outputs['caption'], batch_first=True, padding_value=padding_value
        )

    return {
        "images": torch.stack(outputs['image']),
        "captions": captions
    }


def build_dataloader(dataset, batch_size, num_workers, shuffle, ddp=False, pad_token_id=0):
    sampler = distributed.DistributedSampler(dataset, shuffle=shuffle) if ddp else RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn_warpper(pad_token_id),
            worker_init_fn=seed_worker
            )


def get_dataset(config, transform, modes):
    
    captions = pd.read_csv(config.cap_data_path)
    tokenizer = Tokenizer(config.vocab_size, captions)

    if len(modes) > 1:
        dataset = CustomDataset(config, captions, tokenizer, transform)
        train_size = int(len(dataset) * 0.9)
        valid_size = len(dataset) - train_size
        dataset = random_split(dataset, [train_size, valid_size])
    else:
        test_indices = random.sample(range(len(captions)), int(len(captions) * 0.002))
        captions = captions.iloc[test_indices].reset_index(drop=True)
        dataset = [CustomDataset(config, captions, tokenizer, transform)]

    return {mode:ds for mode, ds in zip(modes, dataset)}, tokenizer


def get_dataloader(config, transform):
    """
    Returns:
        (Dict[phase: DataLoader]): dataloader for training
    Examples:
        {'train': DataLoader, 'valid': DataLoader}
        {'test': DataLoader}
    """
    n_gpu = torch.cuda.device_count()
    n_cpu = os.cpu_count()
    num_workers = min([4 * n_gpu, config.batch_size // n_gpu, config.batch_size // n_cpu])  # number of workers
    modes = ['train', 'valid'] if config.mode == 'train' else ['test']

    dict_dataset, tokenizer = get_dataset(config, transform, modes)

    dataloader = {mode: build_dataloader(dict_dataset[mode], config.batch_size, num_workers, mode == 'train', config.ddp, tokenizer.pad_token_id) for mode in modes}

    return dataloader, tokenizer