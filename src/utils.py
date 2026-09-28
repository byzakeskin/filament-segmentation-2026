import torch
from torch.utils.data import random_split, DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_val_loaders(dataset, val_ratio=0.15, batch_size=2, seed=42):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    return train_loader, val_loader