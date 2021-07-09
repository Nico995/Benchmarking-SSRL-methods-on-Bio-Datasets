import glob
import os
from datetime import datetime

import torchvision
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import Kather
from methods import train_supervised, val_supervised
from models.downstream_classification import DownstreamClassification
from models.supervised import Supervised
from parser import load_args
from utils import dataset_name, config as cf
from utils.custom_exceptions import *

from scripts import main_loop

''' 
Since this project focuses on benchmarking different training methods and different dataset, we need modularity. 
To avoid endless chains of if-else, we will make great use of the dictionary structures below. 
We will assign a python object (dataset, training loop, ...) to each possible value a user can input in the args.
'''


# TODO: write checkpointing just for last fc (classifier) in case of downstream task

def main():
    """
    This is the entry point of the whole project. In this main function all the basic structures for the training get
    initialized.
    """

    # Load command line arguments
    args = load_args()

    # Initialize data_loader structures
    try:
        ext = cf.image_extension_by_dataset[dataset_name(args.data)]
        ds_train = cf.dataset[dataset_name(args.data)](args, ext=ext, get_indices=False)
        ds_val = cf.dataset[dataset_name(args.data)](args, mode='val', ext=ext, get_indices=False)
    except KeyError:
        raise DatasetNotSupportedError(dataset_name(args.data))

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, num_workers=args.threads, drop_last=True, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.val_batch_size, num_workers=args.threads, drop_last=True, shuffle=True)

    # if available
    try:
        num_classes = cf.classes_by_method[args.method]
        model = cf.model_by_method[args.method](num_classes=num_classes, version=args.version)

        # Get training method
        train_ = cf.train_by_method[args.method]
        val_ = cf.val_by_method[args.method]
        criterion = cf.criterion_by_method[args.method]

    except KeyError:
        raise MethodNotSupportedError(args.method)

    model = model.cuda()

    # Load optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Load learning rate scheduler
    lr_scheduler = MultiStepLR(optimizer, [80, 100], 0.1)

    # Create folder to save checkpoints
    checkpoint_folder = f"{args.level}-{dataset_name(args.data)}-{args.method}_" + \
                        datetime.now().strftime(f"%Y-%m-%d_%H:%M:%S")
    os.makedirs(os.path.join(args.checkpoint_path, checkpoint_folder), exist_ok=True)
    print("saving data at ", os.path.join(args.checkpoint_path, checkpoint_folder))

    # Tensorboard writer
    # writer = SummaryWriter(comment=f'_{args.method}_{args.epochs}ep_{args.batch_size}bs_{args.data.split("/")[-2]}')
    writer = SummaryWriter(logdir=os.path.join(args.checkpoint_path, checkpoint_folder))

    main_loop(args, model, train_, val_, dl_train, dl_val, optimizer, lr_scheduler, criterion, writer, checkpoint_folder)


if __name__ == '__main__':
    main()