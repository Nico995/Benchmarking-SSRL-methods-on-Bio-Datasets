import torch

from transforms import DiscreteRandomRotation
from utils.metrics import accuracy, corrects


def train_rotation(model, img, lbl, optimizer, criterion):
    model.train()

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
    corr = corrects(out, angle)

    # Compute parameter's gradient
    loss.backward()

    # Back-propagate and update parameters
    optimizer.step()

    return loss.item(), corr.item()


def val_rotation(model, img, lbl, criterion):
    model.eval()

    # Initialize rotation transform
    rot = DiscreteRandomRotation([0, 90, 180, 270])

    # Perform a rotation for each image
    img, angle = map(list, zip(*[tuple(rot(i)) for i in img]))

    # Convert lists back to tensors
    img, angle = torch.stack(img, dim=0).cuda(), torch.tensor(angle).cuda()

    with torch.no_grad():
        # Get model prediction
        out = model(img)

        # Compute loss & metrics
        loss = criterion(out, angle)
        corr = corrects(out, angle)

    return loss.item(), corr.item()
