from functools import partial

import torch
from torch.utils import model_zoo
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops.misc import FrozenBatchNorm2d


def fasterrcnn_shape_resnet50(device, num_classes):
    # url_resnet50_trained_on_SIN = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar'
    url_resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'

    checkpoint = model_zoo.load_url(url_resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN)

    # Some magic to rename the keys so that it loads as resnet_fpn_backbone
    state_dict_body = dict([('.'.join(['body'] + k.split('.')[1:]), v) for k, v in checkpoint["state_dict"].items()])

    # This is to resolve the issue of NANs coming up in training.
    # See 
    FBN = partial(FrozenBatchNorm2d, eps=1E-5)
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, norm_layer=FBN, trainable_layers=3).cuda()

    missing, unexpected = backbone.load_state_dict(state_dict_body, strict=False)

    print('When creating shape-resnet50...\nMissing states: {}\nUnexpected states: {}'.format(missing, unexpected))

    model = FasterRCNN(backbone, num_classes=2)
    model.to(device)

    params_to_update = model.parameters()
    optimizer = torch.optim.SGD(params_to_update, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.5)

    return model, optimizer, lr_scheduler


def fasterrcnn_resnet101(device, trainable_layers, box_nms_thresh, num_classes):
    return fasterrcnn_resnetx('resnet101', device, trainable_layers, box_nms_thresh, num_classes)


def fasterrcnn_resnet50(device, trainable_layers, box_nms_thresh, num_classes):
    return fasterrcnn_resnetx('resnet50', device, trainable_layers, box_nms_thresh, num_classes)


def fasterrcnn_resnetx(backbone_name, device, trainable_layers,
                       box_nms_thresh,
                       num_classes):
    backbone = resnet_fpn_backbone(backbone_name, pretrained=True, trainable_layers=trainable_layers)
    model = FasterRCNN(backbone, num_classes=num_classes, box_nms_thresh=box_nms_thresh)
    model.to(device)

    params_to_update = model.parameters()
    optimizer = torch.optim.SGD(params_to_update, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.5)

    return model, optimizer, lr_scheduler
