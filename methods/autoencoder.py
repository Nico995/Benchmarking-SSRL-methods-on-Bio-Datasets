def train_autoencoder(model, img, lbl, optimizer, criterion):
    # Zero the parameter's gradient
    optimizer.zero_grad()
    img = img.cuda()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(img, out)

    # Compute parameter's gradient
    loss.backward()

    # Back-propagate and update parameters
    optimizer.step()

    return loss.item(), -1


def val_autoencoder(model, img, lbl, criterion):
    img = img.cuda()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(img, out)
    return loss.item(), -1
