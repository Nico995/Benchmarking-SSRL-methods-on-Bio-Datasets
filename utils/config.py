from torch.nn import CrossEntropyLoss, MSELoss

from dataset import Kather, Pedestrians
from methods import train_rotation, val_rotation, train_jigsaw, val_jigsaw, train_autoencoder, val_autoencoder, \
    train_supervised, val_supervised
from models import Rotation, Jigsaw, AutoEncoder, ImagenetPretrained, RandomInitialization
from models.supervised import Supervised
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
}

image_extension_by_dataset = {
    'kather': 'tif',
    'pedestrians': 'pgm'
}

classes_by_dataset = {
    'kather': 8,
    'pedestrians': 2
}

classes_by_method = {
    'rotation': 4,
    'jigsaw': 100,
    'autoencoder': 0,
    'imagenet_pretrained': classes_by_dataset[dataset_name(args.data)],
    'random_initialization': classes_by_dataset[dataset_name(args.data)],
    'instance_discrimination': 128,
    'supervised': classes_by_dataset[dataset_name(args.data)],
    'downstream': classes_by_dataset[dataset_name(args.data)]
}

train_by_method = {
    'rotation': train_rotation,
    'jigsaw': train_jigsaw,
    'autoencoder': train_autoencoder,
    'imagenet_pretrained': train_supervised,
    'random_initialization': train_supervised,
    'supervised': train_supervised,
    'downstream': train_supervised
}

val_by_method = {
    'rotation': val_rotation,
    'jigsaw': val_jigsaw,
    'autoencoder': val_autoencoder,
    'imagenet_pretrained': val_supervised,
    'random_initialization': val_supervised,
    'supervised': val_supervised,
    'downstream': val_supervised
}

criterion_by_method = {
    'rotation': CrossEntropyLoss(),
    'jigsaw': CrossEntropyLoss(),
    'autoencoder': MSELoss(),
    'imagenet_pretrained': CrossEntropyLoss(),
    'random_initialization': CrossEntropyLoss(),
    'instance_discrimination': CrossEntropyLoss(),
    'supervised': CrossEntropyLoss(),
    'downstream': CrossEntropyLoss(),
}

model_by_method = {
    'rotation': Rotation,
    'jigsaw': Jigsaw,
    'autoencoder': AutoEncoder,
    'imagenet_pretrained': ImagenetPretrained,
    'random_initialization': RandomInitialization,
    'supervised': Supervised,
    'downstream': Supervised,
}