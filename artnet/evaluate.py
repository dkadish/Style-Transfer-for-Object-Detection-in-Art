import os
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import chain
from pathlib import Path

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events
from ignite.handlers import global_step_from_engine

from .data import get_eval_data_loader, configuration_data
from .engines import create_evaluator
from .metrics import CocoAP, CocoAP75, CocoAP5
from .utilities import draw_debug_images, draw_mask, get_model_instance_segmentation, get_iou_types, \
    get_model_instance_detection
from ..utils import utils
from ..utils.coco_utils import convert_to_coco_api


def run(batch_size=1, log_interval=50, debug_images_interval=10,
        val_dataset_ann_file='~/bigdata/coco/annotations/instances_val2017.json', input_checkpoint='',
        log_dir="/tmp/tensorboard_logs", use_mask=True, backbone_name='resnet101'):
    hparam_dict = {
        # 'warmup_iterations': warmup_iterations,
        'batch_size': batch_size,
        # 'test_size': test_size,
        # 'epochs': epochs,
        # 'trainable_layers': trainable_layers,
        # 'load_optimizer': load_optimizer,
        # 'lr': lr,
        # 'momentum': momentum,
        # 'weight_decay': weight_decay,
    }

    # Load the old hparams
    hparam_file = Path(input_checkpoint).parent / 'hparams.pickle'
    try:
        print('Opening hparams file from {}'.format(hparam_file.absolute()))
        with open(hparam_file, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            data = pickle.load(f)
            print('Updating hparams with {}'.format(data))
            hparam_dict.update(data)
    except FileNotFoundError as e:
        print('HParam file not found at {}'.format(hparam_file.absolute()))
    
    print('Params: {}'.format(hparam_dict))

    # Define train and test datasets
    val_loader, labels_enum = get_eval_data_loader(
        val_dataset_ann_file,
        batch_size,
        configuration_data.get('image_size'),
        use_mask=use_mask)
    val_dataset = list(chain.from_iterable(zip(*batch) for batch in iter(val_loader)))
    coco_api_val_dataset = convert_to_coco_api(val_dataset)
    num_classes = max(labels_enum.keys()) + 1  # number of classes plus one for background class
    configuration_data['num_classes'] = num_classes
    print('Testing with {} classes...'.format(num_classes))

    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size

    if use_mask:
        print('Loading MaskRCNN Model...')
        model = get_model_instance_segmentation(num_classes, configuration_data.get('mask_predictor_hidden_layer'))
    else:
        print('Loading FasterRCNN Model...')
        model = get_model_instance_detection(num_classes, backbone_name=backbone_name)
    iou_types = get_iou_types(model)

    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        print("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # copy the model to each device
    model.to(device)

    print('Loading model checkpoint from {}'.format(input_checkpoint))
    input_checkpoint = torch.load(input_checkpoint, map_location=torch.device(device))
    model.load_state_dict(input_checkpoint['model'])

    if use_mask:
        comment = 'mask'
    else:
        comment = 'box-{}'.format(backbone_name)
    tb_logger = TensorboardLogger(log_dir=log_dir, comment=comment)
    writer = tb_logger.writer

    # define Ignite's train and evaluation engine
    evaluator = create_evaluator(model, device)

    coco_ap = CocoAP(coco_api_val_dataset, iou_types)
    coco_ap_05 = CocoAP5(coco_api_val_dataset, iou_types)
    coco_ap_075 = CocoAP75(coco_api_val_dataset, iou_types)
    coco_ap.attach(evaluator, "AP")
    coco_ap_05.attach(evaluator, "AP0.5")
    coco_ap_075.attach(evaluator, "AP0.75")

    tb_logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag='evaluation',
            metric_names=['AP', 'AP0.5', 'AP0.75'],
            global_step_transform=global_step_from_engine(evaluator)
        ),
        event_name=Events.COMPLETED
    )

    @evaluator.on(Events.STARTED)
    def on_evaluation_started(engine):
        model.eval()
        # engine.state.coco_evaluator = CocoEvaluator(coco_api_val_dataset, iou_types)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def on_eval_iteration_completed(engine):
        images, targets, results = engine.state.output
        if engine.state.iteration % log_interval == 0:
            print("Evaluation: Iteration: {}".format(engine.state.iteration))

        if engine.state.iteration % debug_images_interval == 0:
            print('Saving debug image...')
            for n, debug_image in enumerate(draw_debug_images(images, targets, results)):
                writer.add_image("evaluation/image_{}_{}".format(engine.state.iteration, n),
                                 debug_image, evaluator.state.iteration, dataformats='HWC')
                if 'masks' in targets[n]:
                    writer.add_image("evaluation/image_{}_{}_mask".format(engine.state.iteration, n),
                                     draw_mask(targets[n]), evaluator.state.iteration, dataformats='HW')
                    curr_image_id = int(targets[n]['image_id'])
                    writer.add_image("evaluation/image_{}_{}_predicted_mask".format(engine.state.iteration, n),
                                     draw_mask(results[curr_image_id]).squeeze(), evaluator.state.iteration,
                                     dataformats='HW')
        images = targets = results = engine.state.output = None

    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        # gather the stats from all processes
        # engine.state.coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        # engine.state.coco_evaluator.accumulate()
        # engine.state.coco_evaluator.summarize()
        #
        # pr_50, pr_75 = get_pr_levels(engine.state.coco_evaluator.coco_eval['bbox'])
        # plot_pr_curve_tensorboard(pr_50, pr_75, writer=writer)
        print('Writing hparams: {}'.format(hparam_dict))

        writer.add_hparams(hparam_dict=hparam_dict, metric_dict={
            'hparams/AP': coco_ap.ap,
            'hparams/AP.5': coco_ap_05.ap5,
            'hparams/AP.75': coco_ap_075.ap75
        })

        coco_ap.write_tensorboard_pr_curve(writer)

    # evaluator.state = State()
    evaluator.run(val_loader)
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training and validation')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--debug_images_interval', type=int, default=10,
                        help='how many batches to wait before logging debug images')
    parser.add_argument('--val_dataset_ann_file', type=str, default='./annotations/instances_val2017.json',
                        help='annotation file of test dataset')
    parser.add_argument('--input_checkpoint', type=str, default='',
                        help='Loading model weights from this checkpoint.')
    parser.add_argument("--log_dir", type=str, default="./runs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--use_mask", default=False, type=bool,
                        help='use MaskRCNN if True. If False, use FasterRCNN for boxes only.')
    parser.add_argument("--backbone_name", type=str, default='resnet101',
                        help='which backbone to use. options are resnet101, resnet50, and shape-resnet50')
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        utils.mkdir(args.log_dir)

    run(**dict(args._get_kwargs()))
