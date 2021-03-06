from glob import glob
from os.path import join

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage, RandomCrop, CenterCrop, Resize, RandomHorizontalFlip


class Pedestrians(torch.utils.data.Dataset):
    """
    Custom dataset class to manage images and labels
    """

    # TODO: Implement get indices
    def __init__(self, args, mode='train', ext='tif', get_indices=False):
        super(Pedestrians, self).__init__()

        self.data = args.data
        self.mode = mode
        self.ext = ext

        self.rescale_size = args.rescale_size
        self.crop_size = args.crop_size

        self.images = []
        self.labels = []

        # Load images and labels (according to self.mode.upper() being [TRAIN/TEST/VAL])
        for subdir in glob(join(self.data, f'*_{self.mode.upper()}')):
            label = int(subdir.split('/')[-1].split('_')[0][-1])
            images_files = glob(join(subdir, f"*.{self.ext}"))
            self.images.extend(images_files)
            self.labels.extend([label, ] * len(images_files))

        # initialize data transformations
        self.trans = {
            'train': Compose([
                Resize(self.rescale_size),
                RandomCrop(self.crop_size),
                RandomHorizontalFlip(p=0.5),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            'val': Compose([
                Resize(self.rescale_size),
                CenterCrop(self.crop_size),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        }

        self.totensor = ToTensor()
        self.toimage = ToPILImage()

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = torch.repeat_interleave(self.totensor(image), 3, 0)
        # Repeat Gray image on 3 channels

        return self.trans[self.mode](image), self.labels[index]

    def __len__(self):
        return len(self.images)
