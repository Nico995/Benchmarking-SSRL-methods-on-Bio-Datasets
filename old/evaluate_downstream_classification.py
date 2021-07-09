import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import tqdm

from methods import val_supervised
from models.supervised import Supervised

from torch.utils.data import DataLoader

import utils.config as cf
from parser import load_args
from utils import dataset_name
from utils.custom_exceptions import *

''' 
Since this project focuses on benchmarking different training methods and different dataset, we need modularity. 
To avoid endless chains of if-else, we will make great use of the dictionary structures below. 
We will assign a python object (dataset, training loop, ...) to each possible value a user can input in the args.
'''


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
        ds_test = cf.dataset[dataset_name(args.data)](args, mode='test', ext=ext)
        num_classes = cf.classes_by_dataset[dataset_name(args.data)]
    except KeyError:
        raise DatasetNotSupportedError(dataset_name(args.data))

    dl_test = DataLoader(ds_test, batch_size=args.batch_size, num_workers=args.threads, drop_last=True, shuffle=True)

    test_ = val_supervised
    criterion = CrossEntropyLoss()

    try:
        weights = f'../models/pretrained_weights/{dataset_name(args.data)}_{args.method}.pth'
    except KeyError:
        raise DatasetNotSupportedError(dataset_name(args.data))

    model = Supervised(num_classes=num_classes, weights=weights).cuda()
    #########################
    # Validation Batch Loop #
    #########################
    tq = tqdm.tqdm(total=len(dl_test) * args.batch_size)
    tq.set_description(f'downstream task test')

    test_running_loss = []
    test_running_acc = []
    for i, (img, lbl) in enumerate(dl_test):
        img, lbl = img.cuda(), lbl.cuda()
        with torch.no_grad():
            loss, acc = test_(model, img, lbl, criterion)
        test_running_loss.append(loss)
        test_running_acc.append(acc)

    # Update progress bar
    tq.set_postfix({'test_loss': '%.6f' % np.mean(test_running_loss),
                    'test_acc': '%.6f' % np.mean(test_running_acc)})
    tq.close()

    # Save checkpoints


if __name__ == '__main__':
    main()
