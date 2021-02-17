import os
import random
from operator import add
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CocoDetection

from artnet.ignite.transforms import get_transform
from artnet.ignite.utilities import safe_collate

configuration_data = {'image_size': 512, 'mask_predictor_hidden_layer': 256}


class CocoMask(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, use_mask=True):
        super(CocoMask, self).__init__(root, annFile, transforms, target_transform, transform)
        self.transforms = transforms
        self.use_mask = use_mask

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        if len(ann_ids) == 0:
            return None

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # From boxes [x, y, w, h] to [x1, y1, x2, y2]
        new_target = {"image_id": torch.as_tensor(target[0]['image_id'], dtype=torch.int64),
                      "area": torch.as_tensor([obj['area'] for obj in target], dtype=torch.float32),
                      "iscrowd": torch.as_tensor([obj['iscrowd'] for obj in target], dtype=torch.int64),
                      "boxes": torch.as_tensor([obj['bbox'][:2] + list(map(add, obj['bbox'][:2], obj['bbox'][2:]))
                                                for obj in target], dtype=torch.float32),
                      "labels": torch.as_tensor([obj['category_id'] for obj in target], dtype=torch.int64)}
        if self.use_mask:
            mask = [coco.annToMask(ann) for ann in target]
            if len(mask) > 1:
                mask = np.stack(tuple(mask), axis=0)
            new_target["masks"] = torch.as_tensor(mask, dtype=torch.uint8)

        if self.transforms is not None:
            img, new_target = self.transforms(img, new_target)

        return img, new_target


def get_eval_data_loader(test_ann_file, batch_size, image_size, use_mask, num_workers=6):
    train_ann_file = None
    test_size = None
    return get_data_loaders(train_ann_file, test_ann_file, batch_size, test_size, image_size, use_mask,
                            num_workers=num_workers)


def get_data_loaders(train_ann_file, test_ann_file, batch_size, test_size, image_size, use_mask,
                     _use_toy_testing_set=False, num_workers=6, train_set_size=None):
    # first, crate PyTorch dataset objects, for the train and validation data.
    root = Path.joinpath(Path(test_ann_file).parent.parent, test_ann_file.split('_')[1].split('.')[0])
    print('Loading validation image files from {}'.format(root))
    dataset_test = CocoMask(
        root=root,
        annFile=test_ann_file,
        transforms=get_transform(train=False, image_size=image_size),
        use_mask=use_mask)

    labels_enumeration = dataset_test.coco.cats

    if test_size is not None:
        indices_val = torch.randperm(len(dataset_test)).tolist()
        dataset_val = torch.utils.data.Subset(dataset_test, indices_val[:test_size])
    else:
        dataset_val = dataset_test

    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=safe_collate, pin_memory=True)

    if train_ann_file is not None:  # This is just loading a testing set for evaluation, not a training set.
        dataset = CocoMask(
            root=Path.joinpath(Path(train_ann_file).parent.parent, train_ann_file.split('_')[1].split('.')[0]),
            annFile=train_ann_file,
            transforms=get_transform(train=True, image_size=image_size),
            use_mask=use_mask)

        labels_enumeration = dataset.coco.cats

        if _use_toy_testing_set:
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(1000)),
                                      num_workers=num_workers, collate_fn=safe_collate, pin_memory=True)
        elif train_set_size is not None:
            if train_set_size > len(dataset):
                raise ValueError(
                    'The size of the training set ({}) must be smaller than the total size of the dataset ({}).'.format(
                        train_set_size, len(dataset)))
            srs = SubsetRandomSampler(random.sample(range(len(dataset)), k=train_set_size))
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=srs,
                                      num_workers=num_workers,
                                      collate_fn=safe_collate, pin_memory=True)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=safe_collate, pin_memory=True)

        return train_loader, val_loader, labels_enumeration

    return val_loader, labels_enumeration
