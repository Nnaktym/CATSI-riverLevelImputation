import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return None

    def update(self, val: Any, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return None


class TimeSeriesDataSet(Dataset):
    """Time series dataset."""

    def __init__(self, data):
        super().__init__()
        self.content = data

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]


def build_data_loader(
    dataset: TimeSeriesDataSet,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 64,
    shuffle: bool = True,
    testing: bool = False,
) -> Any:
    """Build data loader for the dataset."""

    def pad_time_series_batch(
        batch_data: Any,
    ) -> dict[str, Any]:
        """Pad time series batch data."""
        lengths = [x["length"] for x in batch_data]
        sids = [x["sid"] for x in batch_data]
        lengths, data_idx = torch.sort(
            torch.LongTensor(lengths),
            descending=True,
        )
        batch_data = [batch_data[idx] for idx in data_idx]
        sids = [sids[idx] for idx in data_idx]
        data_dict = {}
        data_dict["values"] = pad_sequence(
            [torch.FloatTensor(x["pt_with_na"]) for x in batch_data], batch_first=True
        ).to(device)
        data_dict["masks"] = pad_sequence(
            [torch.FloatTensor(x["observed_mask"]) for x in batch_data],
            batch_first=True,
        ).to(device)
        data_dict["time_stamps"] = pad_sequence(
            [torch.FloatTensor(x["time_stamps"]) for x in batch_data], batch_first=True
        ).to(device)
        data_dict["rain"] = pad_sequence(
            [torch.FloatTensor(x["rain"]) for x in batch_data], batch_first=True
        ).to(device)
        data_dict["rain_acc"] = pad_sequence(
            [torch.FloatTensor(x["rain_accumulation"]) for x in batch_data],
            batch_first=True,
        ).to(device)
        data_dict["river_mean"] = pad_sequence(
            [torch.FloatTensor(x["river_mean"]) for x in batch_data], batch_first=True
        ).to(device)
        data_dict["river_std"] = pad_sequence(
            [torch.FloatTensor(x["river_std"]) for x in batch_data], batch_first=True
        ).to(device)
        data_dict["lengths"] = lengths.to(device)
        data_dict["sids"] = sids

        if not testing:
            data_dict["evals"] = pad_sequence(
                [torch.FloatTensor(x["pt_ground_truth"]) for x in batch_data], batch_first=True
            ).to(device)
            data_dict["eval_masks"] = pad_sequence(
                [torch.FloatTensor(x["eval_mask"]) for x in batch_data], batch_first=True
            ).to(device)

        return data_dict

    data_iter = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pad_time_series_batch,
    )

    return data_iter
