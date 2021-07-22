from glob import glob
from os.path import join

import numpy as np
import torch.nn
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


class CRC(torch.utils.data.Dataset):

    def __init__(self, args, mode='train', ext='svs', get_indices=False, zoom_level=200, ):
        super(CRC, self).__init__()

        self.data = args.data
        self.mode = mode
        self.ext = ext
        self.get_indices = get_indices
        self.zoom_level = zoom_level

        self.rescale_size = args.rescale_size
        self.crop_size = args.crop_size

        self.images = []
        self.labels = []

        # Load images and labels (according to self.mode.upper() being [TRAIN/TEST/VAL])
        for subdir in glob(join(self.data, '*')):
            label = int(subdir.split('/')[-1].split('_')[0][-1])
            images_files = glob(join(subdir, f"*.{self.ext}"))

            self.images.extend(images_files)
            self.labels.extend([label, ] * len(images_files))

        # initialize data transformations
        self.trans = {
            'train': Compose([
                ToTensor(),
                # Normalize((0.650, 0.472, 0.584), (0.256, 0.327, 0.268)),
            ]),
            'val': Compose([
                ToTensor(),
                # Normalize((0.650, 0.472, 0.584), (0.256, 0.327, 0.268)),
            ])
        }

    def __getitem__(self, index):
        # Open the whole-slide file
        slide = OpenSlide(self.images[index])

        # Initialize a deepzoom object witht he size of the desired tile (tile_size+2*overlap should be a power of 2)
        data_gen = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=True)
        zoom_level = min(data_gen.level_count - 1, self.zoom_level)

        # Choose a zoom level and get the tiles available for said level
        # Note that 0 and last index are not included -> - [1, 1]
        # Note that the last tile could not be a full tipe -> - [1, 1]
        tiles_number = np.array(data_gen.level_tiles[zoom_level]) - [1, 1] - [1, 1]

        # Get the total number of available tiles by multiplying row tiles by column tiles
        tot_tiles = np.array(tiles_number).prod()

        # Get as many random tiles in that range as self.tiles_per_image indicates (sum 1 because the lib starts from 1)
        random_tile_index = np.random.randint(tot_tiles)

        # Go back to (row, col) notation
        random_tile = (random_tile_index % tiles_number[0] + 1, random_tile_index // tiles_number[0] + 1)

        tile = data_gen.get_tile(zoom_level, random_tile)
        # tile = np.array(data_gen.get_tile(zoom_level, random_tile))
        # tile = torch.tensor(tile)

        if self.get_indices:
            return self.trans.get(self.mode, self.trans['val'])(tile), index
        return self.trans.get(self.mode, self.trans['val'])(tile), self.labels[index]

    def __len__(self):
        return len(self.images)
