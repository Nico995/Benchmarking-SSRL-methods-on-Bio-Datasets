import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

from utils.metrics import accuracy
from utils import batch_to_plottable_image
from transforms import DiscreteRandomRotation


class Scrambler(nn.Module):
    """
    Splits image into a grid and extracts some pseudo-random permutation
    """

    def __init__(self, permutations=100):
        self.perms = np.load('data/permutations/naroozi_perms_100_patches_9_max.npy')
        exit()


def train_jigsaw(model, img, lbl, optimizer, criterion):
    pass
