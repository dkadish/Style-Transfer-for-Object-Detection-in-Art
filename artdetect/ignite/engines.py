import math
import copy
from pprint import pprint

import torch

from ignite.engine import Engine
from ..utils import utils


def create_trainer(model, device):
    def update_model(engine, batch):
        images, targets = copy.deepcopy(batch)
        images_model, targets_model = prepare_batch(batch, device=device)

        # print('Images: ')
        # for i, img in enumerate(images_model):
        #     print('Image {}: {}'.format(i, type(img)))
        #     print('Image {}: {}'.format(i, img.shape))
        #
        # print('Targets: ')
        # for i, tgt in enumerate(targets_model):
        #     print('Target {}: {}'.format(i, type(tgt)))
        #     print('Target {}: {}'.format(i, tgt.shape))

        loss_dict = model(images_model, targets_model)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        engine.state.optimizer.zero_grad()
        if not math.isfinite(loss_value):
            print("Loss is {}, resetting loss and skipping training iteration".format(loss_value))
            print('Loss values were: ', loss_dict_reduced)
            print('Input labels were: ', [target['labels'] for target in targets])
            print('Input boxes were: ', [target['boxes'] for target in targets])
            loss_dict_reduced = {k: torch.tensor(0) for k, v in loss_dict_reduced.items()}
        else:
            losses.backward()
            engine.state.optimizer.step()

        if engine.state.warmup_scheduler is not None:
            engine.state.warmup_scheduler.step()

        images_model = targets_model = None

        return images, targets, loss_dict_reduced
    return Engine(update_model)


def create_evaluator(model, device):
    def update_model(engine, batch):
        try:
            images, targets = prepare_batch(batch, device=device)
            images_model = copy.deepcopy(images)

            try:
                torch.cuda.synchronize()
            except AssertionError:
                pass

            with torch.no_grad():
                outputs = model(images_model)

            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

            # print('Evaluating iou_types: {}'.format(engine.state.coco_evaluator.iou_types))
            # engine.state.coco_evaluator.update(res)

        except ValueError as e:
            print('Warning. Empty batch. Returning empty images, targets lists.')
            images, targets, res = [], [], {}

        images_model = outputs = None

        return images, targets, res
    return Engine(update_model)


def prepare_batch(batch, device=None):
    images, targets = batch
    images = list(image.to(device, non_blocking=True) for image in images)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

    # Remove degenerate targets
    for target_idx, target in enumerate(targets):
        boxes = target["boxes"]
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            to_remove = degenerate_boxes.any(dim=1).nonzero().view(-1)
            to_keep = set(range(len(boxes))) - set(to_remove.tolist())
            boxes = boxes[list(to_keep)]
            target["boxes"] = boxes

            print('Found {} degenerate boxes. Keeping {} boxes.'.format(len(to_remove.tolist()), len(list(to_keep))))
            pprint(boxes)

    # print('New batch: {} images, {} targets.'.format(len(images), len(targets)))

    return images, targets

import tensorboard.program
