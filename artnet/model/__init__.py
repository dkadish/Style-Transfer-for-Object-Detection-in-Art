# from abc import ABCMeta, abstractmethod
#
# import torch
# import torchvision
# from torch.utils import model_zoo
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
#
#
# class ModelBase(metaclass=ABCMeta):
#     @abstractmethod
#     def init_lr_scheduler(self):
#         pass
#
#     @abstractmethod
#     def init_optimizer(self):
#         pass
#
#     @abstractmethod
#     def init_params(self):
#         pass
#
#     @abstractmethod
#     def init_model(self):
#         pass
#
#     @abstractmethod
#     def init_roi_pooler(self):
#         pass
#
#     @abstractmethod
#     def init_anchor_generator(self):
#         pass
#
#     @abstractmethod
#     def init_backbone(self):
#         pass
#
#     @abstractmethod
#     def init_features(self):
#         pass
#
#     @abstractmethod
#     def init_num_classes(self):
#         pass
#
#     @abstractmethod
#     def init_device(self):
#         pass
#
#
# class FasterRCNN_ResNet101(ModelBase):
#     def __init__(self):
#         self.device = self.init_device()
#
#         # our dataset has two classes only - background and person
#         self.num_classes = self.init_num_classes()
#
#         # load a pre-trained model for classification and return
#         # only the features
#         self.features = self.init_features()
#         self.backbone = self.init_backbone()
#
#         # let's make the RPN generate 5 x 3 anchors per spatial
#         # location, with 5 different sizes and 3 different aspect
#         # ratios. We have a Tuple[Tuple[int]] because each feature
#         # map could potentially have different sizes and
#         # aspect ratios
#         # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#         #                                    aspect_ratios=((0.5, 1.0, 2.0),))
#         self.anchor_generator = self.init_anchor_generator()
#
#         # let's define what are the feature maps that we will
#         # use to perform the region of interest cropping, as well as
#         # the size of the crop after rescaling.
#         # if your backbone returns a Tensor, featmap_names is expected to
#         # be [0]. More generally, the backbone should return an
#         # OrderedDict[Tensor], and in featmap_names you can choose which
#         # feature maps to use.
#         self.roi_pooler = self.init_roi_pooler()
#
#         # put the pieces together inside a FasterRCNN model
#         self.model = self.init_model()
#
#         # # construct an optimizer
#         self.params = self.init_params()
#         self.optimizer = self.init_optimizer()
#         # optimizer = torch.optim.SGD(params, lr=0.002,
#         #                             momentum=0.9, weight_decay=0.0005)
#
#         # # and a learning rate scheduler which decreases the learning rate by
#         # # 10x every 3 epochs
#         self.lr_scheduler = self.init_lr_scheduler()
#         #  gamma=0.1)
#
#     def init_lr_scheduler(self):
#         return torch.optim.lr_scheduler.StepLR(self.optimizer,
#                                                step_size=3,
#                                                gamma=0.5)
#
#     def init_optimizer(self):
#         return torch.optim.SGD(self.params, lr=0.005,
#                                momentum=0.9, weight_decay=0.0005)
#
#     def init_params(self):
#         return [p for p in self.model.parameters() if p.requires_grad]
#
#     def init_model(self):
#         model = FasterRCNN(self.backbone,
#                            num_classes=self.num_classes,
#                            rpn_anchor_generator=self.anchor_generator,
#                            box_roi_pool=self.roi_pooler)
#
#         # # move model to the right device
#         model.to(self.device)
#
#         return model
#
#     def init_roi_pooler(self):
#         return torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],
#                                                   output_size=7,
#                                                   sampling_ratio=2)
#
#     def init_anchor_generator(self):
#         return AnchorGenerator(sizes=((128, 256, 512),),
#                                aspect_ratios=((0.5, 1.0, 2.0),))
#
#     def init_backbone(self):
#         backbone = torch.nn.Sequential(*self.features)
#
#         # FasterRCNN needs to know the number of
#         # output channels in a backbone. For mobilenet_v2, it's 1280
#         # so we need to add it here.
#         # For resnet101, I THINK it is 1000
#         backbone.out_channels = 1024
#
#         return backbone
#
#     def init_features(self):
#
#         features = list(torchvision.models.resnet101(pretrained=True).children())[:-3]
#
#         for parameters in [feature.parameters() for i, feature in enumerate(features) if i <= 4]:
#             for parameter in parameters:
#                 parameter.requires_grad = False
#
#         return features
#
#     def init_num_classes(self):
#         return 2
#
#     def init_device(self):
#         return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
#
# class FasterRCNN_ResNet50_SIN(FasterRCNN_ResNet101):
#     def init_features(self):
#         url_resnet50_trained_on_SIN = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar'
#
#         backbone_model = torchvision.models.resnet50(pretrained=False).cuda()
#         # backbone_model = torch.nn.DataParallel(backbone_model).cuda()
#         checkpoint = model_zoo.load_url(url_resnet50_trained_on_SIN)
#
#         # Some magic to rename the keys so that it doens't have to be parallelized
#         state_dict = dict([('.'.join(k.split('.')[1:]), v) for k, v in checkpoint["state_dict"].items()])
#
#         backbone_model.load_state_dict(state_dict)
#
#         features = list(backbone_model.children())[:-3]
#
#         return features
