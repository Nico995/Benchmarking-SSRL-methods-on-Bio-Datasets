import torch
from utils.metrics import accuracy, corrects


def train_supervised(model, img, lbl, optimizer, criterion):
    # Zero the parameter's gradient
    optimizer.zero_grad(set_to_none=True)

    # Move data to GPU
    img, lbl = img.cuda(), lbl.cuda()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(out, lbl)
    corr = corrects(out, lbl)

    # Compute parameter's gradient
    loss.backward()

    # Back-propagate and update parameters
    optimizer.step()

    return loss.item(), corr.item()


def val_supervised(model, img, lbl, criterion):
    # Move data to GPU
    img, lbl = img.cuda(), lbl.cuda()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(out, lbl)
    corr = corrects(out, lbl)

    return loss.item(), corr.item()
