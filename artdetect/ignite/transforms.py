import random
import torch
from PIL import Image

from torchvision.transforms import functional as F


def get_transform(train, image_size):
    transforms = [Resize(size=(image_size, image_size)), ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class Resize(object):
    """Resize the input PIL image to given size.
    If boxes is not None, resize boxes accordingly.
    Args:
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
      random_interpolation: (bool) randomly choose a resize interpolation method.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    Example:
    >> img, boxes = resize(img, boxes, 600)  # resize shorter side to 600
    >> img, boxes = resize(img, boxes, (500,600))  # resize image size to (500,600)
    >> img, _ = resize(img, None, (500,600))  # resize image only
    """
    def __init__(self, size, max_size=1000, random_interpolation=False):
        self.size = size
        self.max_size = max_size
        self.random_interpolation = random_interpolation

    def __call__(self, image, target):
        """Resize the input PIL image to given size.
        If boxes is not None, resize boxes accordingly.
        Args:
          image: (PIL.Image) image to be resized.
          target: (tensor) object boxes, sized [#obj,4].
        """
        w, h = image.size
        if isinstance(self.size, int):
            size_min = min(w, h)
            size_max = max(w, h)
            sw = sh = float(self.size) / size_min
            if sw * size_max > self.max_size:
                sw = sh = float(self.max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = self.size
            sw = float(ow) / w
            sh = float(oh) / h

        method = random.choice([
            Image.BOX,
            Image.NEAREST,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
            Image.BILINEAR]) if self.random_interpolation else Image.BILINEAR
        image = image.resize((ow, oh), method)
        if target is not None and "masks" in target:
            resized_masks = torch.nn.functional.interpolate(
                input=target["masks"][None].float(),
                size=(512, 512),
                mode="nearest",
            )[0].type_as(target["masks"])
            target["masks"] = resized_masks
        if target is not None and "boxes" in target:
            resized_boxes = target["boxes"] * torch.tensor([sw, sh, sw, sh])
            target["boxes"] = resized_boxes
        return image, target
