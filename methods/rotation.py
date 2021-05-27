from matplotlib import pyplot as plt

from utils.metrics import accuracy
from utils import batch_to_plottable_image
from transforms import DiscreteRandomRotation

import torch


def train_rotation(model, img, lbl, optimizer, criterion):
    # Zero the parameter's gradient
    optimizer.zero_grad()

    # Initialize rotation transform
    rot = DiscreteRandomRotation([0, 90, 180, 270])
    # Perform a rotation for each image
    img, angle = map(list, zip(*[tuple(rot(i)) for i in img]))

    # Convert lists back to tensors
    img = torch.stack(img, dim=0).cuda()
    angle = torch.tensor(angle).cuda()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(out, angle)
    acc = accuracy(angle, out)

    # Compute parameter's gradient
    loss.backward()

    # Back-propagate and update parameters
    optimizer.step()

    return loss.item(), acc.item()


def val_rotation(model, img, lbl, criterion):
    # Initialize rotation transform
    rot = DiscreteRandomRotation([0, 90, 180, 270])
    # Perform a rotation for each image
    img, angle = map(list, zip(*[tuple(rot(i)) for i in img]))

    # Convert lists back to tensors
    img = torch.stack(img, dim=0).cuda()
    angle = torch.tensor(angle).cuda()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(out, angle)
    acc = accuracy(angle, out)

    return loss.item(), acc.item()
