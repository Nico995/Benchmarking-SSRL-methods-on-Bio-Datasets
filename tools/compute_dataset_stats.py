import argparse
from glob import glob
from os.path import join
from os import makedirs
import numpy as np
from PIL import Image
import tqdm
import random
import shutil
from glob import glob
from os.path import join

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/kather/', help='data path')
    args = parser.parse_args()

    images_files = []
    # Get all image files
    for subdir in glob(join(args.data, '*')):
        images_files.extend(glob(join(subdir, '*.tif')))

    images = []
    for image_file in tqdm.tqdm(images_files):
        images.append(np.array(Image.open(image_file))/255)
    images = np.stack(images, axis=0)

    print(images.shape)
    print('mean ', images.mean(axis=(0, 1, 2)))
    print('std ', images.std(axis=(0, 1, 2)))
