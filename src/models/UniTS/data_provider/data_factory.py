# Import dataloader from main directory
import sys  # nopep8
import os  # nopep8
base_path = os.getenv("BASE_PATH")  # nopep8
sys.path.append(os.path.join(base_path, "src/data"))  # nopep8
from FeaturesDataset import FeaturesUniTS  # nopep8

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


def random_subset(dataset, pct, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, idx[:int(len(dataset) * pct)].long().numpy())


def data_provider(args, config, flag, ddp=False):  # args,
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if 'anomaly_detection' in config['task_name']:  # working on one gpu
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if 'anomaly_detection' in config['task_name']:
        drop_last = False
        win_size = args.win_size

        data_set = FeaturesUniTS(
            dataset=config['data'],
            win_size=win_size,
            flag=flag,
            k=args.k,
            source=args.source,
            confounding=args.confounding,
            step=args.step,
        )
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        print("ddp mode is set to false for anomaly_detection", ddp, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
