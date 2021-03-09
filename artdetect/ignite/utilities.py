import json
from functools import partial

import attr
import cv2
import numpy as np
import torch
import torchvision
from torch.utils import model_zoo
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.transforms import functional as F

from ..utils import utils


def safe_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return utils.collate_fn(batch)


def draw_boxes(im, boxes, labels, color=(150, 0, 0)):
    for box, draw_label in zip(boxes, labels):
        draw_box = box.astype('int')
        im = cv2.rectangle(im, tuple(draw_box[:2]), tuple(draw_box[2:]), color, 2)
        im = cv2.putText(im, str(draw_label), (draw_box[0], max(0, draw_box[1]-5)),
                         cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
    return im


def draw_debug_images(images, targets, predictions=None, score_thr=0.3):
    debug_images = []
    for image, target in zip(images, targets):
        img = draw_boxes(np.array(F.to_pil_image(image.cpu())),
                         [box.cpu().numpy() for box in target['boxes']],
                         [label.item() for label in target['labels']])
        if predictions:
            img = draw_boxes(img,
                             [box.cpu().numpy() for box, score in
                              zip(predictions[target['image_id'].item()]['boxes'],
                                  predictions[target['image_id'].item()]['scores']) if score >= score_thr],
                             [label.item() for label, score in
                              zip(predictions[target['image_id'].item()]['labels'],
                                  predictions[target['image_id'].item()]['scores']) if score >= score_thr],
                             color=(0, 150, 0))
        debug_images.append(img)
    return debug_images


def draw_mask(target):
    masks = [channel*label for channel, label in zip(target['masks'].cpu().numpy(), target['labels'].cpu().numpy())]
    masks_sum = sum(masks)
    masks_out = masks_sum + 25*(masks_sum > 0)
    return (masks_out*int(255/masks_out.max())).astype('uint8')

# def draw_boxes(target):
#     #FIXME THIS!
#     masks = [channel*label for channel, label in zip(target['masks'].cpu().numpy(), target['labels'].cpu().numpy())]
#     masks_sum = sum(masks)
#     masks_out = masks_sum + 25*(masks_sum > 0)
#     return (masks_out*int(255/masks_out.max())).astype('uint8')


def get_model_instance_segmentation(num_classes, hidden_layer, pretrained=True, trainable_backbone_layers=3):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, trainable_backbone_layers=trainable_backbone_layers)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_model_instance_detection(num_classes, backbone_name='resnet101', pretrained_backbone=True, trainable_layers=3):
    if backbone_name == 'shape-resnet50':
        url_resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'
        
        try:
            checkpoint = model_zoo.load_url(url_resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN)
        except RuntimeError as e:
            checkpoint = model_zoo.load_url(url_resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN, map_location=torch.device('cpu'))

        # Some magic to rename the keys so that it loads as resnet_fpn_backbone
        state_dict_body = dict(
            [('.'.join(['body'] + k.split('.')[1:]), v) for k, v in checkpoint["state_dict"].items()])

        # This is to resolve the issue of NANs coming up in training.
        # See
        fbn = partial(FrozenBatchNorm2d, eps=1E-5)

        try:
            backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone, norm_layer=fbn, trainable_layers=trainable_layers).cuda()
        except (RuntimeError, AssertionError) as e:
            backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone, norm_layer=fbn, trainable_layers=trainable_layers).cpu()

        missing, unexpected = backbone.load_state_dict(state_dict_body, strict=False)
        print('When creating shape-resnet50...\nMissing states: {}\nUnexpected states: {}'.format(missing, unexpected))
    else:
        backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained_backbone, trainable_layers=trainable_layers)
    model = FasterRCNN(backbone, num_classes=num_classes)

    return model

def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")

    print('iou_types: {}'.format(iou_types))
    return iou_types


@attr.s(auto_attribs=True)
class CocoLikeAnnotations():
    def __attrs_post_init__(self):
        self.coco_like_json: dict = {'images': [], 'annotations': []}
        self._ann_id: int = 0

    def update_images(self, file_name, height, width, id):
        self.coco_like_json['images'].append({'file_name': file_name,
                                         'height': height, 'width': width,
                                         'id': id})

    def update_annotations(self, box, label_id, image_id, is_crowd=0):
        segmentation, bbox, area = self.extract_coco_info(box)
        self.coco_like_json['annotations'].append({'segmentation': segmentation, 'bbox': bbox, 'area': area,
                                              'category_id': int(label_id), 'id': self._ann_id, 'iscrowd': is_crowd,
                                              'image_id': image_id})
        self._ann_id += 1

    @staticmethod
    def extract_coco_info(box):
        segmentation = list(map(int, [box[0], box[1], box[0], box[3], box[2], box[3], box[2], box[1]]))
        bbox = list(map(int, np.append(box[:2], (box[2:] - box[:2]))))
        area = int(bbox[2] * bbox[3])
        return segmentation, bbox, area

    def dump_to_json(self, path_to_json='/tmp/inference_results/inference_results.json'):
        with open(path_to_json, "w") as write_file:
            json.dump(self.coco_like_json, write_file)
