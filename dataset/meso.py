import xml.etree.ElementTree as et
from glob import glob
from os.path import join

import numpy as np
import torch.nn
from openslide import OpenSlide
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor


class MESO(torch.utils.data.Dataset):

    def __init__(self, args, mode='train', ext='svs', get_indices=False, zoom_level=200, oversample=True, fix_tiles=True):
        super(MESO, self).__init__()

        self.data = args.data
        self.mode = mode
        self.ext = ext
        self.get_indices = get_indices
        self.zoom_level = zoom_level
        self.oversample = oversample
        self.fix_tiles = fix_tiles
        self.batch_size = args.train_batch_size
        self.tile_size = args.crop_size  # TODO: correct wrong naming "crop" to "tile"

        self.rescale_size = args.rescale_size
        self.crop_size = args.crop_size

        self.images = []
        self.labels = []
        self.xmls = []

        # Load images and labels (according to self.mode.upper() being [TRAIN/TEST/VAL])
        for subdir in glob(join(self.data, f'*_{self.mode.upper()}')):
            label = int(subdir.split('/')[-1].split('_')[0][-1])
            images_files = glob(join(subdir, f"*.{self.ext}"))
            xml = glob(join(subdir, f"*.xml"))
            self.images.extend(images_files)
            self.labels.extend([label, ] * len(images_files))
            self.xmls.extend(xml)

        self.images = sorted(self.images)
        self.labels = sorted(self.labels)
        self.xmls = sorted(self.xmls)

        # If we want always the same tiles from the image, then build a pseudo random vector of crop of seeds
        # To be applied before extracting a random crop from a wsi
        if self.fix_tiles:
            self.tile_seeds = np.random.randint(self.__len__(), size=self.__len__())

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
        wrapped_index = index % len(self.images)

        # Parse the xml file and get the list of annotations as the first element of the root
        tree = et.parse(self.xmls[wrapped_index])
        root = tree.getroot()
        annotations = root[0]

        bboxes = []
        # for each ROI
        for path in annotations:
            coordinates = path[0]

            # Get the coordinate in integer
            bbox = [(int(coord.get('X').split('.')[0]), int(coord.get('Y').split('.')[0]))
                    for coord in coordinates]

            # Keep only rois that allow a 224 by 224 crop
            if bbox[1][0] - bbox[0][0] > 224:
                if bbox[2][1] - bbox[1][1] > 224:
                    bboxes.append(bbox)

        # Open the whole-slide file
        slide = OpenSlide(self.images[wrapped_index])

        # TODO: we're always taking maximum zoom level (0 in ndpi files), we should implement something smarter
        zoom_level = 0

        # Get random ROI
        if self.fix_tiles:
            np.random.seed(self.tile_seeds[index])

        roi_idx = np.random.choice(len(bboxes))
        roi = bboxes[roi_idx]

        # Make sure we start from a position that allows the extraction of 224 sized tiles
        all_cols = [pos[0] for pos in roi]
        all_rows = [pos[1] for pos in roi]

        col_range = (np.min(all_cols), np.max(all_cols) - self.tile_size)
        row_range = (np.min(all_rows), np.max(all_rows) - self.tile_size)

        # Get a random number between start and end, and sum it to start
        col_idx = np.random.choice(col_range[1] - col_range[0])
        col = col_range[0] + col_idx

        # Same as above
        row_idx = np.random.choice(row_range[1] - row_range[0])
        row = row_range[0] + row_idx

        # Crop the desired tile at max zoom level
        tile = slide.read_region((col, row), size=(224, 224), level=0).convert('RGB')

        if self.get_indices:
            # TODO: Implement a way of getting a unique index that takes into account wrapping and random tiling
            return self.trans.get(self.mode, self.trans['val'])(tile), index
        else:
            return self.trans.get(self.mode, self.trans['val'])(tile), self.labels[wrapped_index]
            # return self.trans.get(self.mode, self.trans['val'])(tile), self.labels[wrapped_index], index

    def __len__(self):
        if self.oversample:
            return self.batch_size * 10
        else:
            return len(self.images)
