import torch


def train_instance_discrimination(model, img, lbl, optimizer, criterion):
    # Zero the parameter's gradient
    optimizer.zero_grad()

    # Move data to GPU
    img, lbl = img.cuda(), lbl.cuda()

    # Get model prediction
    out = model(img, lbl)

    # Compute loss & metrics
    loss = criterion(out, lbl)

    # Compute parameter's gradient
    loss.backward()

    # Back-propagate and update parameters
    optimizer.step()

    # Optimization purposes
    loss_item = loss.item()
    return loss_item, loss_item


def val_instance_discrimination(model, img, lbl, criterion):
    model.eval()

    # Move data to GPU
    img, lbl = img.cuda(), lbl.cuda()

    with torch.no_grad():
        # Get model prediction
        out = model(img, lbl)

        # Compute loss & metrics
        loss = criterion(out, lbl)

    # Optimization purposes
    loss_item = loss.item()

    return loss_item, loss_item
