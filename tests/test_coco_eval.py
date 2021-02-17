from itertools import chain
from pathlib import Path
from unittest import TestCase

from artnet.ignite.data import get_eval_data_loader
from artnet.utils.coco_eval import CocoEvaluator
from artnet.utils.coco_utils import convert_to_coco_api


class TestCocoEvaluator(TestCase):

    def setUp(self) -> None:
        val_dataset_ann_file = (Path(
            __file__) / '..' / '..' / 'data' / 'PeopleArt-Coco' / 'annotations' / 'peopleart_val.json').resolve()
        batch_size = 2
        iou_types = ["bbox"]
        val_loader, labels_enum = get_eval_data_loader(
            str(val_dataset_ann_file),
            batch_size,
            512,
            use_mask=False)
        val_dataset = list(chain.from_iterable(zip(*batch) for batch in iter(val_loader)))
        coco_api_val_dataset = convert_to_coco_api(val_dataset)

        self.coco_eval = CocoEvaluator(coco_api_val_dataset, iou_types)

    def test_synchronize_between_processes(self):

        self.coco_eval.synchronize_between_processes()

        # self.fail()
