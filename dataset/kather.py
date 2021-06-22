from glob import glob
from os.path import join

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, CenterCrop, Resize, RandomHorizontalFlip


class Kather(torch.utils.data.Dataset):
    """
    Custom dataset class to manage images and labels
    """

    def __init__(self, args, mode='train', ext='tif', get_indices=False):
        super(Kather, self).__init__()

        self.data = args.data
        self.mode = mode
        self.ext = ext
        self.get_indices = get_indices

        self.rescale_size = args.rescale_size
        self.crop_size = args.crop_size

        self.images = []
        self.labels = []

        # Load images and labels (according to self.mode.upper() being [TRAIN/TEST/VAL])
        for subdir in glob(join(self.data, f'*_{self.mode.upper()}')):
            label = int(subdir.split('/')[-1].split('_')[0][-1]) - 1
            images_files = glob(join(subdir, f"*.{self.ext}"))
            self.images.extend(images_files)
            self.labels.extend([label, ] * len(images_files))

        # initialize data transformations
        self.trans = {
            'train': Compose([
                Resize(self.rescale_size),
                RandomCrop(self.crop_size),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # Normalize((0.650, 0.472, 0.584), (0.256, 0.327, 0.268))
            ]),
            'val': Compose([
                Resize(self.rescale_size),
                CenterCrop(self.crop_size),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                # Normalize((0.650, 0.472, 0.584), (0.256, 0.327, 0.268))
            ])
        }

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.get_indices:
            return self.trans.get(self.mode, self.trans['val'])(image), index
        return self.trans.get(self.mode, self.trans['val'])(image), self.labels[index]

    def __len__(self):
        return len(self.images)
