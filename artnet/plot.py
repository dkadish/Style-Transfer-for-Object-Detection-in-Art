import numpy as np
import matplotlib as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from pycocotools.cocoeval import COCOeval


def get_pr_levels(ce: COCOeval):
    all_precision = ce.eval['precision']

    # pr = all_precision[:, :, 0, 0, 2]  # data for IoU@.50:.05:.95
    pr_50 = all_precision[0, :, 0, 0, 2]  # data for IoU@0.5
    pr_75 = all_precision[5, :, 0, 0, 2]  # data for IoU@0.75

    return pr_50, pr_75


def plot_pr_curve_tensorboard(p50, p75, writer=None, write_averages=False):
    if writer is None:
        writer = SummaryWriter()

    for x, y in zip(range(101), p50):
        writer.add_scalar('pr_curve/AP.5', y, global_step=x)

    for x, y in zip(range(101), p75):
        writer.add_scalar('pr_curve/AP.75', y, global_step=x)

    if write_averages:
        writer.add_scalar('metrics/AP.5', np.mean(p50), 0)
        writer.add_scalar('metrics/AP.5', np.mean(p75), 0)