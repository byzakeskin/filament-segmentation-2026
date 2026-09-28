import math
import time

import torch


def _to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [
        {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
        for t in targets
    ]
    return images, targets


def train_one_epoch(model, optimizer, loader, device, scaler,
                    epoch=0, max_iters=None, print_every=20):
    model.train()
    running, n = 0.0, 0
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        if max_iters is not None and i >= max_iters:
            break

        images, targets = _to_device(images, targets, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Loss sonsuz/NaN oldu (iter {i}): {loss_dict}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        n += 1

        if (i + 1) % print_every == 0:
            sec_per_it = (time.time() - t0) / (i + 1)
            print(f"[epoch {epoch}] iter {i + 1}/{len(loader)}  "
                  f"loss {running / n:.4f}  ({sec_per_it:.2f} sn/iter)")

    total = time.time() - t0
    return running / max(n, 1), total / max(n, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device, max_iters=None):

    model.train()
    total, n = 0.0, 0
    for i, (images, targets) in enumerate(loader):
        if max_iters is not None and i >= max_iters:
            break
        images, targets = _to_device(images, targets, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
        n += 1
    return total / max(n, 1)