# Style Transfer for Object Detection in Art

1. Create the dataset
1. Train the model
1. Evaluate the model

### Dependencies
This code is tested with
* PyTorch 1.6
* Torchvision 0.7
* PyTorch Ignite 0.4.2

It may work with updated versions, but this is not guaranteed!

## Train a Model

```
python -m artnet.train

usage: train.py [-h] [--train_dataset_ann_file TRAIN_DATASET_ANN_FILE] [--val_dataset_ann_file VAL_DATASET_ANN_FILE] [--input_checkpoint INPUT_CHECKPOINT] [--output_dir OUTPUT_DIR] [--log_dir LOG_DIR] [--log_interval LOG_INTERVAL]
                [--debug_images_interval DEBUG_IMAGES_INTERVAL] [--record_histograms RECORD_HISTOGRAMS] [--use_mask USE_MASK] [--backbone_name BACKBONE_NAME] [--trainable_layers TRAINABLE_LAYERS] [--test_size TEST_SIZE]
                [--use_toy_testing_data USE_TOY_TESTING_DATA] [--warmup_iterations WARMUP_ITERATIONS] [--lr LR] [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--step_size STEP_SIZE] [--gamma GAMMA]
                [--early_stopping EARLY_STOPPING] [--patience PATIENCE] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--load_optimizer LOAD_OPTIMIZER] [--load_params LOAD_PARAMS] [--num_workers NUM_WORKERS]
                [--train_set_size TRAIN_SET_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --step_size STEP_SIZE
                        step size for learning scheduler (default: 3)
  --gamma GAMMA         gamma for learning scheduler (default: 0.1)
  --early_stopping EARLY_STOPPING
                        use the early stopping function (default: False)
  --patience PATIENCE   early stopping patience setting (number of epochs to keep going after decline (default: 3)

input / output:
  --train_dataset_ann_file TRAIN_DATASET_ANN_FILE
                        annotation file of train dataset (default: ./annotations/instances_train2017.json)
  --val_dataset_ann_file VAL_DATASET_ANN_FILE
                        annotation file of test dataset (default: ./annotations/instances_val2017.json)
  --input_checkpoint INPUT_CHECKPOINT
                        Loading model weights from this checkpoint. (default: )
  --output_dir OUTPUT_DIR
                        output directory for saving models checkpoints (default: ./checkpoints)
  --log_dir LOG_DIR     log directory for Tensorboard log output (default: ./runs)

logs:
  --log_interval LOG_INTERVAL
                        how many batches to wait before logging training status (default: 100)
  --debug_images_interval DEBUG_IMAGES_INTERVAL
                        how many batches to wait before logging debug images (default: 50)
  --record_histograms RECORD_HISTOGRAMS
                        save histograms during training (default: True)

network:
  --use_mask USE_MASK   use MaskRCNN if True. If False, use FasterRCNN for boxes only. (default: False)
  --backbone_name BACKBONE_NAME
                        which backbone to use. options are resnet101, resnet50, and shape-resnet50 (default: resnet101)
  --trainable_layers TRAINABLE_LAYERS
                        number of layers to train (1-5) (default: 3)

evaluation:
  --test_size TEST_SIZE
                        number of frames from the test dataset to use for validation (default: 2000)
  --use_toy_testing_data USE_TOY_TESTING_DATA
                        use a small toy dataset to make sure things work (default: False)

learning:
  --warmup_iterations WARMUP_ITERATIONS
                        Number of iteration for warmup period (until reaching base learning rate) (default: 5000)
  --lr LR               learning rate for optimizer (default: 0.005)
  --momentum MOMENTUM   momentum for optimizer (default: 0.9)
  --weight_decay WEIGHT_DECAY
                        weight decay for optimizer (default: 0.0005)

training:
  --batch_size BATCH_SIZE
                        input batch size for training and validation (default: 4)
  --epochs EPOCHS       number of epochs to train (default: 10)
  --load_optimizer LOAD_OPTIMIZER
                        Use optimizer and lr_scheduler saved in the input checkpoint to resume training (default: False)
  --load_params LOAD_PARAMS
                        Use hparameters from the saved pickle file to resume training (default: False)
  --num_workers NUM_WORKERS
                        number of workers to use for data loading (default: 6)
  --train_set_size TRAIN_SET_SIZE
                        number of images in the training data (default: None)
```

## Evaluate a Model

```
python -m artnet.evaluate -h

usage: evaluate.py [-h] [--batch_size BATCH_SIZE] [--log_interval LOG_INTERVAL] [--debug_images_interval DEBUG_IMAGES_INTERVAL] [--val_dataset_ann_file VAL_DATASET_ANN_FILE] [--input_checkpoint INPUT_CHECKPOINT] [--log_dir LOG_DIR]
                   [--use_mask USE_MASK] [--backbone_name BACKBONE_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        input batch size for training and validation (default: 1)
  --log_interval LOG_INTERVAL
                        how many batches to wait before logging training status (default: 50)
  --debug_images_interval DEBUG_IMAGES_INTERVAL
                        how many batches to wait before logging debug images (default: 10)
  --val_dataset_ann_file VAL_DATASET_ANN_FILE
                        annotation file of test dataset (default: ./annotations/instances_val2017.json)
  --input_checkpoint INPUT_CHECKPOINT
                        Loading model weights from this checkpoint. (default: )
  --log_dir LOG_DIR     log directory for Tensorboard log output (default: ./runs)
  --use_mask USE_MASK   use MaskRCNN if True. If False, use FasterRCNN for boxes only. (default: False)
  --backbone_name BACKBONE_NAME
                        which backbone to use. options are resnet50, resnet101, resnet152, and shape-resnet50 (default: resnet152)
```