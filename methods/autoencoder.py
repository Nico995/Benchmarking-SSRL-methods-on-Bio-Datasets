import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage


def visualize(img, out):
    img = ToPILImage()(img[0])
    out = ToPILImage()(out[0])

    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(img)
    ax[1].imshow(out)
    plt.tight_layout()
    ax[0].axis('off')
    ax[0].set_title('original')
    ax[1].axis('off')
    ax[1].set_title('encoded')
    plt.show()


visualized = False


def train_autoencoder(model, img, lbl, optimizer, criterion):
    # Zero the parameter's gradient
    optimizer.zero_grad()
    img = img.cuda()

    # Get model prediction
    out = model(img)

    # Compute loss & metrics
    loss = criterion(out, img)

    # Compute parameter's gradient
    loss.backward()


    # Back-propagate and update parameters
    optimizer.step()

    # Optimization purposes
    loss_item = loss.item()

    # temp variable needed to only visualize the first image of the validation loop
    global visualized
    visualized = False

    return loss_item, loss_item


def val_autoencoder(model, img, lbl, criterion):
    model.eval()

    img = img.cuda()

    with torch.no_grad():
        # Get model prediction
        out = model(img)

        # Compute loss & metrics
        loss = criterion(img, out)

    global visualized
    if not visualized:
        visualized = True

        visualize(img, out)

    # Optimization purposes
    loss_item = loss.item()
    return loss_item, loss_item
