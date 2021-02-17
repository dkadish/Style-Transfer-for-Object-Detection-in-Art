import os
import glob
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser
from pathlib import Path

from .utilities import get_iou_types, draw_boxes, get_model_instance_segmentation, CocoLikeAnnotations, \
    get_model_instance_detection
from ..utils import utils
from torchvision.transforms import functional as F

from PIL import Image
from .transforms import get_transform

# from trains import Task
# task = Task.init(project_name='Object Detection with TRAINS, Ignite and TensorBoard',
#                  task_name='Inference with trained model')


def rescale_box(box, image_size, orig_height, orig_width):
    rescale_height = float(orig_height) / image_size
    rescale_width = float(orig_width) / image_size
    box[:2] *= rescale_width
    box[2:] *= rescale_height
    return box


def run(batch_size=4, detection_thresh=0.4, log_interval=100, debug_images_interval=500,
        input_dataset_root='/media/dan/bigdata/datasets/coco/2017/val2017',
        input_checkpoint='/tmp/checkpoints/model_epoch_10.pth', output_dir="/tmp/inference_results",
        log_dir="/tmp/tensorboard_logs",
        use_mask=True, backbone_name='resnet101'):
    if use_mask:
        comment = 'mask-eval'
    else:
        comment = 'box-{}-eval'.format(backbone_name)
    writer = SummaryWriter(log_dir=log_dir, comment=comment)
    input_checkpoint = torch.load(input_checkpoint)
    labels_enum = input_checkpoint.get('labels_enumeration')
    model_configuration = input_checkpoint.get('configuration')
    model_weights = input_checkpoint.get('model')
    image_size = model_configuration.get('image_size')

    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size

    num_classes = model_configuration.get('num_classes')
    if use_mask:
        print('Loading MaskRCNN Model...')
        model = get_model_instance_segmentation(num_classes, configuration_data.get('mask_predictor_hidden_layer'))
    else:
        print('Loading FasterRCNN Model...')
        model = get_model_instance_detection(num_classes, backbone_name=backbone_name)

    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        print("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # copy the model to each device
    model.to(device)

    # Define train and test datasets
    iou_types = get_iou_types(model)
    use_mask = True if "segm" in iou_types else False

    # Load pretrained model weights
    model.load_state_dict(model_weights)

    # set the model to inference mode
    model.eval()

    images_paths = []
    for file_type in ('*.png', '*.jpg', '*.jpeg'):
        images_paths.extend(glob.glob(os.path.join(input_dataset_root, file_type)))

    transforms = get_transform(train=False, image_size=image_size)

    path_to_json = os.path.join(output_dir, "inference_results.json")
    coco_like_anns = CocoLikeAnnotations()
    batch_images = []
    batch_paths = []
    batch_shapes = []

    for i, image_path in enumerate(images_paths):
        img = Image.open(image_path).convert('RGB')
        batch_shapes.append({'height': img.height, 'width': img.width})
        img, __ = transforms(img)
        batch_images.append(img)
        batch_paths.append(image_path)
        if len(batch_images) < batch_size:
            continue

        input_images = torch.stack(batch_images)

        with torch.no_grad():
            torch_out = model(input_images.to(device))

        for img_num, image in enumerate(input_images):
            valid_detections = torch_out[img_num].get('scores') >= detection_thresh
            img_boxes = torch_out[img_num].get('boxes')[valid_detections].cpu().numpy()
            img_labels_ids = torch_out[img_num].get('labels')[valid_detections].cpu().numpy()
            img_labels = [labels_enum[label]['name'] for label in img_labels_ids]
            image_id = (i + 1 - batch_size + img_num)
            orig_height = batch_shapes[img_num].get('height')
            orig_width = batch_shapes[img_num].get('width')

            coco_like_anns.update_images(file_name=Path(batch_paths[img_num]).name,
                                         height=orig_height, width=orig_width,
                                         id=image_id)

            for box, label, label_id in zip(img_boxes, img_labels, img_labels_ids):
                orig_box = rescale_box(image_size=image_size, orig_height=orig_height, orig_width=orig_width, box=box.copy())
                coco_like_anns.update_annotations(box=orig_box, label_id=label_id,
                                                  image_id=image_id)

            if ((i+1)/batch_size) % log_interval == 0:
                print('Batch {}: Saving detections of file {} to {}'.format(int((i+1)/batch_size),
                                                                            Path(batch_paths[img_num]).name,
                                                                            path_to_json))

            if ((i+1)/batch_size) % debug_images_interval == 0:
                debug_image = draw_boxes(np.array(F.to_pil_image(image.cpu())), img_boxes, img_labels, color=(0, 150, 0))
                writer.add_image("inference/image_{}".format(img_num), debug_image, ((i+1)/batch_size),
                                 dataformats='HWC')

        batch_images = []
        batch_paths = []

    coco_like_anns.dump_to_json(path_to_json=path_to_json)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training and validation (default: 4)')
    parser.add_argument('--detection_thresh', type=float, default=0.4,
                        help='Inference confidence threshold')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--debug_images_interval', type=int, default=500,
                        help='how many batches to wait before logging debug images')
    parser.add_argument('--input_dataset_root', type=str,
                        default='/media/dan/bigdata/datasets/coco/2017/val2017',
                        help='annotation file of test dataset')
    parser.add_argument('--input_checkpoint', type=str, default='/tmp/checkpoints/model_epoch_10.pth',
                        help='Checkpoint to use for inference')
    parser.add_argument("--output_dir", type=str, default="/tmp/inference_results",
                        help="output directory for saving models checkpoints")
    parser.add_argument("--log_dir", type=str, default="/tmp/tensorboard_logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--use_mask", default=False, type=bool,
                        help='use MaskRCNN if True. If False, use FasterRCNN for boxes only.')
    parser.add_argument("--backbone_name", type=str, default='resnet101',
                        help='which backbone to use. options are resnet101, resnet50, and shape-resnet50')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        utils.mkdir(args.output_dir)
    if not os.path.exists(args.log_dir):
        utils.mkdir(args.log_dir)

    run(**dict(args._get_kwargs()))
