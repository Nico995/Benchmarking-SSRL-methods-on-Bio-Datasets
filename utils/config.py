import os

from torch.nn import CrossEntropyLoss, BCELoss

from dataset import Kather, Pedestrians, CRC, MESO
from methods import train_rotation, val_rotation, train_jigsaw, val_jigsaw, train_autoencoder, val_autoencoder, \
    train_supervised, val_supervised, train_instance_discrimination, val_instance_discrimination
from methods.gan import train_gan, val_gan
from models import Rotation, Jigsaw, AutoEncoder, ImagenetPretrained, RandomInitialization, InstanceDiscrimination, \
    Supervised
from models.pretext.gan import GAN
from parser import load_args
from utils import dataset_name

'''
Since this project focuses on benchmarking different training methods and different dataset, we need modularity. 
To avoid endless chains of if-else, we will make great use of the dictionary structures below. 
We will assign a python object (dataset, training loop, ...) to each possible value a user can input in the args.
'''

# Load command line arguments
args = load_args()

# These names must correspond to the name of the dataset root folder in the filesystem
dataset = {
    'kather': Kather,
    'pedestrians': Pedestrians,
    'crc': CRC,
    'meso': MESO,
}

image_extension_by_dataset = {
    'kather': 'tif',
    'pedestrians': 'pgm',
    'crc': 'svs',
    'meso': 'ndpi',
}

classes_by_dataset = {
    'kather': 8,
    'pedestrians': 2,
    'crc': 9,
    'meso': 2,
}

classes_by_method = {
    'rotation': 4,
    'jigsaw': 100,
    'autoencoder': 8,
    'gan': 2,
    'imagenet_pretrained': classes_by_dataset[dataset_name(args.data)],
    'random_initialization': classes_by_dataset[dataset_name(args.data)],
    'instance_discrimination': [128, 3000],
    'supervised': classes_by_dataset[dataset_name(args.data)],
}

train_by_method = {
    'rotation': train_rotation,
    'jigsaw': train_jigsaw,
    'autoencoder': train_autoencoder,
    'gan': train_gan,
    'imagenet_pretrained': train_supervised,
    'random_initialization': train_supervised,
    'instance_discrimination': train_instance_discrimination,
    'supervised': train_supervised,
}

val_by_method = {
    'rotation': val_rotation,
    'jigsaw': val_jigsaw,
    'autoencoder': val_autoencoder,
    'gan': val_gan,
    'imagenet_pretrained': val_supervised,
    'random_initialization': val_supervised,
    'instance_discrimination': val_instance_discrimination,
    'supervised': val_supervised,
}

criterion_by_method = {
    'rotation': CrossEntropyLoss(),
    'jigsaw': CrossEntropyLoss(),
    'autoencoder': BCELoss(),
    'gan': CrossEntropyLoss(),
    'imagenet_pretrained': CrossEntropyLoss(),
    'random_initialization': CrossEntropyLoss(),
    'instance_discrimination': CrossEntropyLoss(),
    'supervised': CrossEntropyLoss(),
}

model_by_method = {
    'rotation': Rotation,
    'jigsaw': Jigsaw,
    'autoencoder': AutoEncoder,
    'gan': GAN,
    'imagenet_pretrained': ImagenetPretrained,
    'random_initialization': RandomInitialization,
    'instance_discrimination': InstanceDiscrimination,
    'supervised': Supervised,
}

# This is the default weight path, weights are only needed when running downstream linear class. training
default_weights_path = \
    None if args.level == 'pretext' else os.path.join(f'../models/pretrained/',
                                                      f'{dataset_name(args.data)}-{args.method}', 'latest.pth')

# Since not every method requires pretrained weights
# (e.g. random init doesn't need to run pretext training first), We only return for some methods
weights_by_method = {
    'rotation': default_weights_path,
    'jigsaw': default_weights_path,
    'autoencoder': default_weights_path,
    'gan': default_weights_path,
    'imagenet_pretrained': None,
    'random_initialization': None,
    'instance_discrimination': default_weights_path,
    'supervised': default_weights_path,
}
