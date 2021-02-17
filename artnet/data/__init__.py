import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Sized

import torch
import torch.utils.data
from PIL import Image
from torch.utils.data.dataset import random_split
from torchvision.datasets import VisionDataset, VOCDetection


class WeightMixin(Sized):

    def __init__(self):
        self._positive = None
        self._negative = None

    def weights(self, overall=1.0):
        '''Gets weights for samples so that positive and negative samples are drawn evenly.'''
        p = overall * (1.0 - len(self.positive) / len(self))
        n = overall * (1.0 - len(self.negative) / len(self))

        w = list([0 for _ in range(len(self))])
        for i in self.positive:
            w[i] = p
        for i in self.negative:
            w[i] = n

        return w

    @property
    def positive(self):
        if self._positive is None:
            self._calculate_positive_and_negative()

        return self._positive

    @property
    def negative(self):
        if self._negative is None:
            self._calculate_positive_and_negative()

        return self._negative

    def _calculate_positive_and_negative(self):
        self._negative = []
        self._positive = []

        for i, (_, target) in enumerate(self):
            if target['boxes'].numel() > 0:
                self._positive.append(i)
            else:
                self._negative.append(i)


class ArtNetDataset(VisionDataset, WeightMixin):

    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, source='artnet',
                 image_set=None) -> None:
        VisionDataset.__init__(self, root, transform=transform, transforms=transforms,
                               target_transform=target_transform)
        WeightMixin.__init__(self)

        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted((Path(self.root) / "JPEGImages").iterdir()))
        # self.annotations = list(sorted((Path(self.root) / "Annotations").iterdir()))
        self.source = source

        if image_set is None:
            self.imgs = list((self._root / 'JPEGImages').glob('**/*.jpg'))
        else:
            with open((self._root / 'ImageSets' / 'Main' / image_set).with_suffix('.txt')) as f:
                file_names = [x.strip() for x in f.readlines()]

            self.imgs = [(self._root / 'JPEGImages' / x).with_suffix('.jpg') for x in file_names]

    def __getitem__(self, idx):
        # Handle slices
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(idx, tuple):
            raise NotImplementedError('Tuple as index')

        # load images and bounding boxes
        img_path = self.imgs[idx]
        rel_path = img_path.relative_to(self._root / 'JPEGImages')

        img = Image.open(img_path).convert("RGB")
        boxes = []

        try:
            annotation_path = next((self._root / 'Annotations').glob(str(rel_path.with_suffix('.*'))))
            # annotation_path = self.annotations[idx]
            tree = ET.parse(Path(annotation_path))
            tree.getroot()
            root = tree.getroot()
            objects = root.findall('object')
            for obj in objects:
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bbox = (xmin, ymin, xmax, ymax)
                boxes.append(bbox)

            # get bounding box coordinates for each mask
            num_objs = len(boxes)

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except (StopIteration, FileNotFoundError, IndexError) as e:  # There's no file there.
            num_objs = 0
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["source"] = self.source

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target  # , image_id

    def __len__(self):
        return len(self.imgs)

    @property
    def _root(self):
        return Path(self.root)


class VOCDetectionSubset(VOCDetection, WeightMixin):

    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None,
                 transforms=None, names=[]):
        VOCDetection.__init__(self, root, year, image_set, download, transform, target_transform, transforms)
        WeightMixin.__init__(self)

        self.names = names

    def __getitem__(self, index):
        img, annotation = super().__getitem__(index)

        annotation['annotation']['object'] = self.filter_annotation_by_object_names(annotation)

        try:
            box_dicts = [o['bndbox'] for o in annotation['annotation']['object']]
            boxes = [(int(b['xmin']), int(b['ymin']), int(b['xmax']), int(b['ymax'])) for b in box_dicts]

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            num_objs = len(boxes)

            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        except IndexError as e:  # There's no file there.
            num_objs = 0
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([index])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        ann = {
            'area': area,
            'boxes': boxes,
            'image_id': image_id,
            'iscrowd': iscrowd,
            'labels': labels,
            'source': 'voc'
        }

        return img, ann

    def filter_annotation_by_object_names(self, annotation):
        return list(filter(lambda o: o['name'] in self.names, annotation['annotation']['object']))


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    @property
    def lengths(self):
        return [len(d) for d in self.datasets]

    @property
    def weights(self):
        return [float(len(self)) / float(l) for l in self.lengths]

    @property
    def weight_list(self):
        w = []
        for i, d in enumerate(self.datasets):
            for _ in range(len(d)):
                w.append(self.weights[i])

        return w


def split_posneg_trainval(datasets: dict, train_fraction=0.7, include_negative=True, use: list = None):
    if use is not None:
        datasets = {k: datasets[k] for k in use}

    datasets_pos = {}
    datasets_neg = {}
    datasets_train = {}
    datasets_validate = {}

    for ds in datasets:
        datasets_pos[ds] = torch.utils.data.Subset(datasets[ds], datasets[ds].positive)
        datasets_neg[ds] = torch.utils.data.Subset(datasets[ds], datasets[ds].negative)

        ds_names, ds_pns = ['{}+', '{}-'], [datasets_pos, datasets_neg]
        if not include_negative:
            ds_names, ds_pns = ['{}+', ], [datasets_pos, ]

        for ds_name, ds_pn in zip(ds_names, ds_pns):
            n_train = int(len(ds_pn[ds]) * train_fraction)
            n_validate = len(ds_pn[ds]) - n_train
            if train_fraction == 1.0:
                datasets_train[ds_name.format(ds)] = ds_pn[ds]
            else:
                datasets_train[ds_name.format(ds)], datasets_validate[ds_name.format(ds)] = random_split(ds_pn[ds],
                                                                                                 [n_train, n_validate])
    if train_fraction == 1.0:
        return datasets_train
    else:
        return datasets_train, datasets_validate
