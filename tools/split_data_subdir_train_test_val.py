import argparse
from glob import glob
from os.path import join
from os import makedirs
import numpy as np
from PIL import Image
import tqdm
import random
import shutil


def split_data(images):

    train_prop = len(images)*60//100
    test_prop = train_prop + len(images)*20//100
    list_imgs = random.sample(images, len(images))

    train_imgs = list_imgs[:train_prop]
    test_imgs = list_imgs[train_prop:test_prop]
    val_imgs = list_imgs[test_prop:]

    return [train_imgs, test_imgs, val_imgs]


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/kather/', help='data path')
    args = parser.parse_args()

    for subdir in glob(join(args.data, "*")):
        # Create train test and val subdirs
        subdir_name = subdir.split('/')[-1]
        subdir_root = join(args.data, subdir_name)
        newdirs = [f"{subdir_root}_TRAIN", f"{subdir_root}_TEST", f"{subdir_root}_VAL"]

        # Split images in 60, 20, 20
        images = list(glob(join(subdir, "*.tif")))
        splits = split_data(images)

        for new_subdir_name, split in zip(newdirs, splits):
            makedirs(new_subdir_name)
            for file in split:
                newfile = join(new_subdir_name, file.split('/')[-1])
                shutil.move(file, newfile)

        shutil.rmtree(subdir)


if __name__ == '__main__':
    main()
