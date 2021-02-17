'''From https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html'''

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


from .engine import train_one_epoch, evaluate
from . import utils
from . import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    return T.Compose(transforms)
