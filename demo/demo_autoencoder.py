import glob
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from models import AutoEncoder

from parser import load_args
from utils import dataset_name, batch_to_plottable_image, config as cf


def main():

    # Load command line arguments
    args = load_args()

    ext = cf.image_extension_by_dataset[dataset_name(args.data)]
    model = AutoEncoder(num_classes=8, version=args.version)
    # print(glob.glob(f'../checkpoints/pretext-{dataset_name(args.data)}-{args.method}_*'))
    weights = os.path.join(f'../models/pretrained/kather-autoencoder/latest.pth')
    model.load_state_dict(torch.load(weights))

    dataset = cf.dataset[dataset_name(args.data)](args, mode='val', ext=ext)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

    for img, lbl in dataloader:
        model.eval()
        out = model(img)
        img, out = batch_to_plottable_image(img), batch_to_plottable_image(out)

        fig, ax = plt.subplots(1, 2)
        ax = ax.ravel()
        ax[0].imshow(img)
        ax[0].set_title("original")
        ax[1].imshow(out)
        ax[1].set_title("autoencoded")
        plt.show()
        exit()


if __name__ == '__main__':
    main()
