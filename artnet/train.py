import copy
import logging
import os
import pickle
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import chain
from operator import itemgetter
from pathlib import Path

import torch
from ignite.contrib.handlers import TensorboardLogger, BasicTimeProfiler
from ignite.contrib.handlers.tensorboard_logger import WeightsHistHandler, OptimizerParamsHandler, OutputHandler
from ignite.engine import Events, State
from ignite.handlers import global_step_from_engine, EarlyStopping

from artnet.ignite.metrics import CocoAP, CocoAP75, CocoAP5
from .data import get_data_loaders, configuration_data
from .engines import create_trainer, create_evaluator
from .utilities import draw_debug_images, draw_mask, get_model_instance_segmentation, get_iou_types, \
    get_model_instance_detection
from ..utils import utils
from ..utils.coco_utils import convert_to_coco_api

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

# task = Task.init(project_name='Object Detection with TRAINS, Ignite and TensorBoard',
#                  task_name='Train MaskRCNN with torchvision')

# configuration_data = task.connect_configuration(configuration_data)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('artnet.ignite.train')
logging.getLogger('ignite.engine.engine.Engine').setLevel(logging.INFO)


def run(warmup_iterations=5000, batch_size=4, test_size=2000, epochs=10, log_interval=100, debug_images_interval=50,
        train_dataset_ann_file='~/bigdata/coco/annotations/instances_train2017.json',
        val_dataset_ann_file='~/bigdata/coco/annotations/instances_val2017.json', input_checkpoint='',
        load_optimizer=False, load_params=False, output_dir="/tmp/checkpoints", log_dir="/tmp/tensorboard_logs",
        lr=0.005, momentum=0.9,
        weight_decay=0.0005, use_mask=True, use_toy_testing_data=False, backbone_name='resnet101', num_workers=6,
        trainable_layers=3, train_set_size=None, early_stopping=False, patience=3, step_size=3, gamma=0.1,
        record_histograms=True):
    # Set the training device to GPU if available - if not set it to CPU
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False  # optimization for fixed input size

    # Write hyperparams
    hparam_dict = {
        'warmup_iterations': warmup_iterations,
        'training_batch_size': batch_size,
        'test_size': test_size,
        'epochs': epochs,
        'trainable_layers': trainable_layers,
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'train_set_size': train_set_size,
        'step_size': step_size,
        'gamma': gamma,
        'early_stopping': early_stopping,
        'patience': patience,
        'total_iterations': 0,
        'total_epochs': 0,
        'timeout': True,
    }

    # Load checkpoint if available
    if input_checkpoint:
        hparam_path = Path(input_checkpoint).parent / 'hparams.pickle'

        logger.info('Loading model checkpoint from '.format(input_checkpoint))
        input_checkpoint = torch.load(input_checkpoint, map_location=torch.device(device))  # FIXME Bad overload

        with open(hparam_path, 'rb') as f:
            hparam_dict = pickle.load(f)

        # Load the training parameters from the saved hparam dictionary
        if load_params:
            warmup_iterations, batch_size, test_size, epochs, trainable_layers, lr, momentum,\
            weight_decay, train_set_size, step_size, gamma, early_stopping, patience = itemgetter(
                'warmup_iterations', 'training_batch_size', 'test_size', 'epochs', 'trainable_layers',
                'lr', 'momentum', 'weight_decay', 'train_set_size', 'step_size', 'gamma', 'early_stopping',
                'patience')(hparam_dict)
            try:
                train_set_size -= 1
            except TypeError as e:
                pass

    print('Hparams: ', hparam_dict)

    # Define train and test datasets
    train_loader, val_loader, labels_enum = get_data_loaders(train_dataset_ann_file,
                                                             val_dataset_ann_file,
                                                             batch_size,
                                                             test_size,
                                                             configuration_data.get('image_size'),
                                                             use_mask=use_mask,
                                                             _use_toy_testing_set=use_toy_testing_data,
                                                             num_workers=num_workers,
                                                             train_set_size=train_set_size)

    # Hparams
    hparam_dict['training_set_size'] = len(train_loader) * batch_size
    hparam_dict['validation_set_size'] = len(val_loader) * batch_size

    with open(os.path.join(output_dir, 'hparams.pickle'), 'wb') as f:
        pickle.dump(hparam_dict, f)

    val_dataset = list(chain.from_iterable(
        zip(*copy.deepcopy(batch)) for batch in iter(val_loader)))  # TODO Figure out what this does and use deepcopy.
    coco_api_val_dataset = convert_to_coco_api(val_dataset)
    num_classes = max(labels_enum.keys()) + 1  # number of classes plus one for background class
    configuration_data['num_classes'] = num_classes

    logger.info('Training with {} classes...'.format(num_classes))

    if use_mask:
        logger.debug('Loading MaskRCNN Model...')
        model = get_model_instance_segmentation(num_classes, configuration_data.get('mask_predictor_hidden_layer'))
    else:
        logger.debug('Loading FasterRCNN Model...')
        model = get_model_instance_detection(num_classes, backbone_name=backbone_name,
                                             trainable_layers=trainable_layers)
    iou_types = get_iou_types(model)

    # if there is more than one GPU, parallelize the model
    if torch.cuda.device_count() > 1:
        logger.debug("{} GPUs were detected - we will use all of them".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # copy the model to each device
    model.to(device)

    if input_checkpoint:
        model.load_state_dict(input_checkpoint['model'])

    logger.debug('Initializing SummaryWriter...')
    if use_mask:
        comment = 'mask'
    else:
        comment = 'box-{}'.format(backbone_name)

    logger.debug('Creating Trainer...')
    # define Ignite's train and evaluation engine
    trainer = create_trainer(model, device)
    logger.debug('Creating Evaluator...')
    evaluator = create_evaluator(model, device)

    logger.debug('Initializing Tensorboard Logger...')
    tb_logger = TensorboardLogger(log_dir=log_dir, comment=comment)
    if record_histograms:
        tb_logger.attach(
            trainer,
            event_name=Events.ITERATION_COMPLETED(every=500),
            log_handler=WeightsHistHandler(model)
        )
    writer = tb_logger.writer

    logger.debug('Setting up profiler...')
    profiler = BasicTimeProfiler()
    profiler.attach(trainer)

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
            global_step_transform=global_step_from_engine(trainer)
        ),
        event_name=Events.EPOCH_COMPLETED
    )

    ## Early stopping
    def score_function(engine):
        ap_score = engine.state.metrics['AP']
        return ap_score

    if early_stopping:
        handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
        evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_intermediate_results():
        logger.debug('Epoch Complete...')
        profiler.print_results(profiler.get_results())

    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        # construct an optimizer
        logger.info('Started Training...')
        params = [p for p in model.parameters() if p.requires_grad]
        engine.state.optimizer = torch.optim.SGD(params,
                                                 lr=lr,
                                                 momentum=momentum,
                                                 weight_decay=weight_decay)

        tb_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(engine.state.optimizer),
            event_name=Events.ITERATION_STARTED
        )

        engine.state.scheduler = torch.optim.lr_scheduler.StepLR(engine.state.optimizer, step_size=step_size,
                                                                 gamma=gamma)
        if input_checkpoint:
            # Load traininer states
            trainer.state.epoch = input_checkpoint['epoch']
            if 'iteration' in input_checkpoint:
                trainer.state.iteration = input_checkpoint['iteration']
            else:
                trainer.state.iteration = int(hparam_dict['training_set_size'] / batch_size * input_checkpoint['epoch'])

            if load_optimizer:
                print('loading optimizer')
                logger.info('Loading optimizer and scheduler...')
                engine.state.optimizer.load_state_dict(input_checkpoint['optimizer'])
                engine.state.scheduler.load_state_dict(input_checkpoint['lr_scheduler'])
                engine.state.scheduler.last_epoch = trainer.state.epoch
            else:
                print('not loading optimizer')


    @trainer.on(Events.EPOCH_STARTED)
    def on_epoch_started(engine):
        logger.debug('Started Epoch...')
        model.train()
        engine.state.warmup_scheduler = None

        #TODO Print optimizer values

        if engine.state.epoch == 1:
            warmup_iters = min(warmup_iterations, len(train_loader) - 1)
            print('Warm up period was set to {} iterations'.format(warmup_iters))
            warmup_factor = 1. / warmup_iters
            engine.state.warmup_scheduler = utils.warmup_lr_scheduler(engine.state.optimizer, warmup_iters,
                                                                      warmup_factor)

    @trainer.on(Events.ITERATION_COMPLETED)
    def on_iteration_completed(engine):
        images, targets, loss_dict_reduced = engine.state.output
        if engine.state.iteration % log_interval == 0:
            loss = sum(loss for loss in loss_dict_reduced.values()).item()
            print("Epoch: {}, Iteration: {}, Loss: {}".format(engine.state.epoch, engine.state.iteration, loss))
            for k, v in loss_dict_reduced.items():
                writer.add_scalar("loss/{}".format(k), v.item(), engine.state.iteration)
            writer.add_scalar("loss/total_loss", sum(loss for loss in loss_dict_reduced.values()).item(),
                              engine.state.iteration)
            # writer.add_scalar("learning_rate/lr", engine.state.optimizer.param_groups[0]['lr'], engine.state.iteration)

        if engine.state.iteration % debug_images_interval == 0:
            for n, debug_image in enumerate(draw_debug_images(images, targets)):
                writer.add_image("training/image_{}".format(n), debug_image, engine.state.iteration, dataformats='HWC')
                if 'masks' in targets[n]:
                    writer.add_image("training/image_{}_mask".format(n),
                                     draw_mask(targets[n]), engine.state.iteration, dataformats='HW')
        images = targets = loss_dict_reduced = engine.state.output = None

    @trainer.on(Events.EPOCH_COMPLETED)
    def on_epoch_completed(engine):
        logger.debug('Finished Epoch...')
        update_hparams(engine)

        engine.state.scheduler.step()
        evaluator.run(val_loader)
        # for res_type in evaluator.state.coco_evaluator.iou_types:
        #     average_precision_05 = evaluator.state.coco_evaluator.coco_eval[res_type].stats[1]
        #     writer.add_scalar("validation-{}/average precision 0_5".format(res_type), average_precision_05,
        #                       engine.state.iteration)
        checkpoint_path = os.path.join(output_dir, 'model_epoch_{}.pth'.format(engine.state.epoch))
        print('Saving model checkpoint')
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': engine.state.optimizer.state_dict(),
            'lr_scheduler': engine.state.scheduler.state_dict(),
            'epoch': engine.state.epoch,
            'iteration': engine.state.iteration,
            'configuration': configuration_data,
            'labels_enumeration': labels_enum}
        utils.save_on_master(checkpoint, checkpoint_path)
        print('Model checkpoint from epoch {} was saved at {}'.format(engine.state.epoch, checkpoint_path))
        checkpoint = None
        evaluator.state = State()

    @trainer.on(Events.COMPLETED)
    def on_training_completed(engine):
        logger.debug('Finished Training...')
        update_hparams(engine, finished=True)

        writer.add_hparams(hparam_dict=hparam_dict, metric_dict={
            'hparams/AP': coco_ap.ap,
            'hparams/AP.5': coco_ap_05.ap5,
            'hparams/AP.75': coco_ap_075.ap75
        })
        logger.debug('Wrote hparams...')

    def update_hparams(engine, finished=False):
        hparam_dict['total_iterations'] = global_step_from_engine(engine)(engine, Events.ITERATION_COMPLETED)
        hparam_dict['total_epochs'] = global_step_from_engine(engine)(engine, Events.EPOCH_COMPLETED)
        hparam_dict['timeout'] = not finished

        if hparam_dict['train_set_size'] is None:
            hparam_dict['train_set_size'] = hparam_dict['training_set_size']

        try:
            shutil.copyfile(os.path.join(output_dir, 'hparams.pickle'),
                            os.path.join(output_dir, 'hparams.pickle.backup'))
            with open(os.path.join(output_dir, 'hparams.pickle'), 'wb') as f:
                pickle.dump(hparam_dict, f)
        except AttributeError as e:
            print('Could not pickle one of the total vars.', e)
            os.replace(os.path.join(output_dir, 'hparams.pickle.backup'), os.path.join(output_dir, 'hparams.pickle'))

    @evaluator.on(Events.STARTED)
    def on_evaluation_started(engine):
        logger.debug('Started Evaluation...')
        model.eval()
        # engine.state.coco_evaluator = CocoEvaluator(coco_api_val_dataset, iou_types)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def on_eval_iteration_completed(engine):
        images, targets, results = engine.state.output
        if engine.state.iteration % log_interval == 0:
            print("Evaluation: Iteration: {}".format(engine.state.iteration))

        if engine.state.iteration % debug_images_interval == 0:
            for n, debug_image in enumerate(draw_debug_images(images, targets, results)):
                print('Drawing debug image "validation/image_{}_{}"'.format(engine.state.iteration, n))
                writer.add_image("evaluation/image_{}_{}".format(engine.state.iteration, n),
                                 debug_image, trainer.state.iteration, dataformats='HWC')
                if 'masks' in targets[n]:
                    writer.add_image("validation/image_{}_{}_mask".format(engine.state.iteration, n),
                                     draw_mask(targets[n]), trainer.state.iteration, dataformats='HW')
                    curr_image_id = int(targets[n]['image_id'])
                    writer.add_image("validation/image_{}_{}_predicted_mask".format(engine.state.iteration, n),
                                     draw_mask(results[curr_image_id]).squeeze(), trainer.state.iteration,
                                     dataformats='HW')
        images = targets = results = engine.state.output = None

    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        logger.debug('Finished Evaluation...')
        # gather the stats from all processes
        # engine.state.coco_evaluator.synchronize_between_processes()
        #
        # # accumulate predictions from all images
        # engine.state.coco_evaluator.accumulate()
        # engine.state.coco_evaluator.summarize()
        #
        # pr_50, pr_75 = get_pr_levels(engine.state.coco_evaluator.coco_eval['bbox'])
        # TODO Bring this back
        # writer.add_hparams(hparam_dict, {
        #     'hparams/AP.5': np.mean(pr_50),
        #     'hparams/AP.75': np.mean(pr_75)
        # })

    logger.debug('Running Trainer...')
    trainer.run(train_loader, max_epochs=epochs)
    writer.close()

    profiler.write_results('{}/time_profiling.csv'.format(output_dir))


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--warmup_iterations', type=int, default=5000,
                        help='Number of iteration for warmup period (until reaching base learning rate)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training and validation')
    parser.add_argument('--test_size', type=int, default=2000,
                        help='number of frames from the test dataset to use for validation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--debug_images_interval', type=int, default=50,
                        help='how many batches to wait before logging debug images')
    parser.add_argument('--train_dataset_ann_file', type=str,
                        default='./annotations/instances_train2017.json',
                        help='annotation file of train dataset')
    parser.add_argument('--val_dataset_ann_file', type=str, default='./annotations/instances_val2017.json',
                        help='annotation file of test dataset')
    parser.add_argument('--input_checkpoint', type=str, default='',
                        help='Loading model weights from this checkpoint.')
    parser.add_argument('--load_optimizer', default=False, type=bool,
                        help='Use optimizer and lr_scheduler saved in the input checkpoint to resume training')
    parser.add_argument('--load_params', default=False, type=bool,
                        help='Use hparameters from the saved pickle file to resume training')
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="output directory for saving models checkpoints")
    parser.add_argument("--log_dir", type=str, default="./runs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate for optimizer")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="weight decay for optimizer")
    parser.add_argument("--use_mask", default=False, type=bool,
                        help='use MaskRCNN if True. If False, use FasterRCNN for boxes only.')
    parser.add_argument("--use_toy_testing_data", default=False, type=bool,
                        help='use a small toy dataset to make sure things work')
    parser.add_argument("--backbone_name", type=str, default='resnet101',
                        help='which backbone to use. options are resnet101, resnet50, and shape-resnet50')
    parser.add_argument("--num_workers", type=int, default=6,
                        help='number of workers to use for data loading')
    parser.add_argument("--trainable_layers", type=int, default=3,
                        help='number of layers to train (1-5)')
    parser.add_argument("--train_set_size", type=int, default=None,
                        help='number of images in the training data')
    parser.add_argument("--early_stopping", default=False, type=bool,
                        help='use the early stopping function')
    parser.add_argument("--patience", type=int, default=3,
                        help='early stopping patience setting (number of epochs to keep going after decline')
    parser.add_argument("--step_size", type=int, default=3,
                        help='step size for learning scheduler')
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="gamma for learning scheduler")
    parser.add_argument("--record_histograms", default=True, type=bool,
                        help='save histograms during training')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        utils.mkdir(args.output_dir)
    if not os.path.exists(args.log_dir):
        utils.mkdir(args.log_dir)

    print(args)

    run(**dict(args._get_kwargs()))
