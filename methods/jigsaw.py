import numpy as np
from matplotlib import pyplot as plt
from itertools import product

import torch
from torch import nn
from torchvision.transforms.functional import crop

from utils.metrics import accuracy
from utils import batch_to_plottable_image
from transforms import DiscreteRandomRotation


# Parameters used to identify non overlapping tiles, for an image of size 64x64
tile_params = {
    "size": 19,
    "space": 21,
    "jitter": 1,
    "offset": 1
}

tiles_start = [i*tile_params['space'] + tile_params['offset'] for i in range(3)]

# 'Good' permutations
perms = np.load('data/permutations/naroozi_perms_100_patches_9_max.npy')


def scramble(tiles):
    """
    Scrambles the batch tiles with some pseudo-random permutations
    Args:
        tiles (torch.tensor): Batch of tiles of shape (T, B, C, W, H)

    Returns:
        (torch.tensor): Batch of scrambled tiles of shape (T, B, C, W, H)
    """

    random_perms = []
    scrambled_tiles = []

    for img in range(tiles.shape[1]):
        # Extract random permutation from list and store it
        random_idx = np.random.choice(perms.shape[0])
        random_perm = perms[random_idx]
        random_perms.append(random_idx)
        # Scramble tiles of image <img> according of said permutation, and store them

        scrambled_tile = torch.stack([tiles[t-1, img] for t in random_perm])
        scrambled_tiles.append(scrambled_tile)

    return torch.stack(scrambled_tiles, dim=1), torch.tensor(np.stack(random_perms))


def tile(batch):
    """
    Generates tiles for the whole batch of images
    Args:
        batch (torch.tensor): Batch of images of shape (T, B, C, W, H)

    Returns:
        (torch.tensor): Batch of tiles of shape (T, B, C, W, H)
    """

    tiles = []

    # for i in image:
    for x, y in list(product(tiles_start, repeat=2)):
        tiles.append(crop(batch, x, y, tile_params['size'], tile_params['size']))
    tiles = torch.stack(tiles)
    return tiles


def visualize_srcamble(img, tiles):
    plt.imshow(batch_to_plottable_image(img))
    plt.show()

    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    ax = ax.ravel()

    for i, t in enumerate(tiles[:, 0, :, :, :]):
        ax[i].imshow(batch_to_plottable_image(t.unsqueeze(0)))

    plt.show()
    exit()


def train_jigsaw(model, img, lbl, optimizer, criterion):
    # Zero the parameter's gradient
    optimizer.zero_grad()

    # Tile the batch of images and scramble them
    tiles = tile(img)
    tiles, permutations = scramble(tiles)

    # Uncomment the line below if you want to check the tile appearance
    # visualize_srcamble(img, tiles)

    tiles, permutations = tiles.cuda(), permutations.cuda()

    # Get model prediction
    out = model(tiles)

    # Compute loss & metrics
    loss = criterion(out, permutations)
    acc = accuracy(permutations, out)

    # Compute parameter's gradient
    loss.backward()

    # Back-propagate and update parameters
    optimizer.step()

    return loss.item(), acc.item()


def val_jigsaw(model, img, lbl, criterion):

    # Tile the batch of images and scramble them
    tiles = tile(img)
    tiles, permutations = scramble(tiles)
    tiles, permutations = tiles.cuda(), permutations.cuda()

    # Get model prediction
    out = model(tiles)

    # Compute loss & metrics
    loss = criterion(out, permutations)
    acc = accuracy(permutations, out)

    return loss.item(), acc.item()
