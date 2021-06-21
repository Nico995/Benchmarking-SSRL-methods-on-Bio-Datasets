from utils.metrics import accuracy


def train_supervised(model, img, lbl, optimizer, criterion):
    # Zero the parameter's gradient
    optimizer.zero_grad()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(out, lbl)
    acc = accuracy(out, lbl)

    # Compute parameter's gradient
    loss.backward()

    # Back-propagate and update parameters
    optimizer.step()

    return loss.item(), acc.item()


def val_supervised(model, img, lbl, criterion):

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(out, lbl)
    acc = accuracy(out, lbl)

    return loss.item(), acc.item()
