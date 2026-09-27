import json
import os
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask


def polygon_to_mask(segmentation, height, width):
    rles = coco_mask.frPyObjects(segmentation, height, width)
    rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
    mask = coco_mask.decode(rle)
    return mask.astype(np.uint8)


class FilamentDataset(Dataset):

    def __init__(self, images_dir, annotations_json, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        with open(annotations_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # image_id -> image bilgisi (file_name, width, height)
        self.images_by_id = {img["id"]: img for img in coco["images"]}

        # image_id -> o gorsele ait tum annotation'lar
        self.annotations_by_image = {}
        for ann in coco["annotations"]:
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        # Sadece en az bir annotation'i olan image_id'leri kullan
        self.image_ids = [
            img_id for img_id in self.images_by_id
            if img_id in self.annotations_by_image
        ]

        self.categories = {c["id"]: c["name"] for c in coco["categories"]}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images_by_id[image_id]

        img_path = self.images_dir / img_info["file_name"]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Gorsel bulunamadi: {img_path}")

        height, width = img_info["height"], img_info["width"]

        anns = self.annotations_by_image[image_id]

        masks, boxes, labels, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            mask = polygon_to_mask(ann["segmentation"], height, width)
            masks.append(mask)

            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        target = {
            "image_id": image_id,
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0

        if self.transforms:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target