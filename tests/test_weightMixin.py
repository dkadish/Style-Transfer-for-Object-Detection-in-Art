from unittest import TestCase

import torch

from artdetect.data import WeightMixin


class WeightMixinTest(WeightMixin):

    def __init__(self):
        WeightMixin.__init__(self)

        self.items = [
            (0, {'boxes': torch.tensor([0, 1, 2, 3])}),
            (1, {'boxes': torch.tensor([])}),
            (2, {'boxes': torch.tensor([])}),
            (3, {'boxes': torch.tensor([0, 1, 2, 3])}),
            (4, {'boxes': torch.tensor([])}),
            (5, {'boxes': torch.tensor([0, 1, 2, 3])}),
            (6, {'boxes': torch.tensor([0, 1, 2, 3])}),
            (17, {'boxes': torch.tensor([])}),
        ]

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)


class TestWeightMixin(TestCase):
    def test_positive_negative(self):
        t = WeightMixinTest()

        assert t.positive == [0, 3, 5, 6]
        assert t.negative == [1, 2, 4, 7]

    def test_weights(self):
        t = WeightMixinTest()

        assert t.weights() == [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert t.weights(overall=0.5) == [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

