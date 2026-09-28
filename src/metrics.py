import numpy as np
import torch


@torch.no_grad()
def _overlaps(pred, gt):
    p = pred.flatten(1).float()
    g = gt.flatten(1).float()
    inter = p @ g.T
    pa, ga = p.sum(1), g.sum(1)
    union = pa[:, None] + ga[None, :] - inter
    iou = inter / union.clamp(min=1)
    return iou, inter, pa, ga


def masks_from_output(output, score_thr=0.5, mask_thr=0.5):
    keep = output["scores"] >= score_thr
    return output["masks"][keep][:, 0] > mask_thr


class PQMeter:

    def __init__(self, iou_thr=0.5):
        self.iou_thr = iou_thr
        self.tp = self.fp = self.fn = 0
        self.iou_sum = 0.0
        self.frag = 0    
        self.merge = 0   

    @torch.no_grad()
    def update(self, pred, gt):
        n, m = len(pred), len(gt)
        if n == 0 or m == 0:
            self.fp += n
            self.fn += m
            return

        iou, inter, pa, ga = _overlaps(pred, gt)
        iou_np = iou.cpu().numpy()
        pairs = np.argwhere(iou_np > self.iou_thr)
        order = np.argsort(-iou_np[pairs[:, 0], pairs[:, 1]])
        used_p, used_g = set(), set()
        tp = 0
        for k in order:
            i, j = pairs[k]
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            tp += 1
            self.iou_sum += float(iou_np[i, j])

        self.tp += tp
        self.fp += n - tp
        self.fn += m - tp

        in_gt = inter / pa.clamp(min=1)[:, None]   
        in_pred = inter / ga.clamp(min=1)[None, :]  
        self.frag += int(((in_gt > 0.5).sum(0) >= 2).sum())
        self.merge += int(((in_pred > 0.5).sum(1) >= 2).sum())

    def compute(self):
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return {
            "PQ": self.iou_sum / denom if denom > 0 else 0.0,
            "SQ": self.iou_sum / self.tp if self.tp > 0 else 0.0,
            "RQ": self.tp / denom if denom > 0 else 0.0,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "fragmented_gt": self.frag, "merged_pred": self.merge,
        }