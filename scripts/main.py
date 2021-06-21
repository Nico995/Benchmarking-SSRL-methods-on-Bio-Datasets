from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from parser import load_args
from utils import dataset_name, config as cf
from utils.custom_exceptions import *

from scripts import main_loop

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
        ds_train = cf.dataset[dataset_name(args.data)](args, ext=ext)
        ds_val = cf.dataset[dataset_name(args.data)](args, mode='val', ext=ext)
    except KeyError:
        raise DatasetNotSupportedError(dataset_name(args.data))

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, num_workers=args.threads, drop_last=True, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.threads, drop_last=True, shuffle=True)

    # if available
    try:
        # Get training method
        train_ = cf.train_by_method[args.method]
        val_ = cf.val_by_method[args.method]

        # Get model
        num_classes = cf.classes_by_method[args.method]
        model = cf.model_by_method[args.method](num_classes=num_classes).cuda()

        # Get pretrained model if evaluating downstream task accuracy
        weights = f'../models/pretrained_weights/{dataset_name(args.data)}_{args.method}.pth'

        # Get criterion
        criterion = cf.criterion_by_method[args.method]
    except KeyError:
        raise MethodNotSupportedError(args.method)

    # Load optimizer
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = Adam(model.parameters(), args.lr)

    # Load learning rate scheduler
    lr_scheduler = MultiStepLR(optimizer, [80, 100], 0.1)

    # Tensorboard writer
    # writer = SummaryWriter(comment=f'_{args.method}_{args.epochs}ep_{args.batch_size}bs_{args.data.split("/")[-2]}')
    writer = SummaryWriter(logdir='dump/asd')  # TODO: remove

    main_loop(args, model, train_, val_, dl_train, dl_val, optimizer, lr_scheduler, criterion, writer)


if __name__ == '__main__':
    main()
