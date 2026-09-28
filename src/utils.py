import random
from collections import defaultdict

from torch.utils.data import Subset, DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def split_by_file(dataset, val_ratio=0.15, seed=42):

    groups = defaultdict(list)
    for idx, image_id in enumerate(dataset.image_ids):
        file_name = dataset.images_by_id[image_id]["file_name"]
        groups[file_name].append(idx)

    file_names = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(file_names)

    n_val_files = int(len(file_names) * val_ratio)
    val_files = set(file_names[:n_val_files])

    train_idx, val_idx = [], []
    for fn, idxs in groups.items():
        (val_idx if fn in val_files else train_idx).extend(idxs)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_train_val_loaders(dataset, val_ratio=0.15, batch_size=2, seed=42):

    train_ds, val_ds = split_by_file(dataset, val_ratio=val_ratio, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    return train_loader, val_loader