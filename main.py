import tqdm
import numpy as np
from tensorboardX import SummaryWriter

from torch.optim import SGD
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from dataset import Kather, Pedestrians
from parser import load_args
from methods import train_rotation, val_rotation, train_jigsaw, val_jigsaw, train_autoencoder, val_autoencoder
from utils.custom_exceptions import *
from utils import dataset_name

from models import Rotation, Jigsaw, AutoEncoder

'''
Since this project focuses on benchmarking different training methods and different dataset, we need modularity. 
To avoid endless chains of if-else, we will make great use of the dictionary structures below. 
We will assign a python object (dataset, training loop, ...) to each possible value a user can input in the args.
'''
# These names must correspond to the name of the dataset root folder in the filesystem
dataset = {
    'kather': Kather,
    'pedestrians': Pedestrians,
}

image_extension_by_dataset = {
    'kather': 'tif',
    'pedestrians': 'pgm'
}

train_by_method = {
    'rotation': train_rotation,
    'jigsaw': train_jigsaw,
    'autoencoder': train_autoencoder,
}

val_by_method = {
    'rotation': val_rotation,
    'jigsaw': val_jigsaw,
    'autoencoder': val_autoencoder,
}

criterion_by_method = {
    'rotation': CrossEntropyLoss(),
    'jigsaw': CrossEntropyLoss(),
    'autoencoder': MSELoss()
}

model_by_method = {
    'rotation': Rotation,
    'jigsaw': Jigsaw,
    'autoencoder': AutoEncoder,
}


def main():
    """
    This is the entry point of the whole project. In this main function all the basic structures for the training get
    initialized.
    """

    # Load command line arguments
    args = load_args()

    # Initialize data_loader structures
    try:
        ext = image_extension_by_dataset[dataset_name(args.data)]
        ds_train = dataset[dataset_name(args.data)](args, ext=ext)
        ds_val = dataset[dataset_name(args.data)](args, mode='val', ext=ext)
    except KeyError:
        raise DatasetNotSupportedError(dataset_name(args.data))

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, num_workers=args.threads, drop_last=True, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.threads, drop_last=True, shuffle=True)

    # if available
    try:
        # Get training method
        train_ = train_by_method[args.method]
        val_ = val_by_method[args.method]
        # Get model
        model = model_by_method[args.method]().cuda()

        # Get criterion
        criterion = criterion_by_method[args.method]
    except KeyError:
        raise MethodNotSupportedError(args.method)

    # Load optimizer
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Load learning rate scheduler
    lr_scheduler = MultiStepLR(optimizer, [80, 100], 0.1)

    # Tensorboard writer
    # writer = SummaryWriter(comment=f'_{args.method}_{args.epochs}ep_{args.batch_size}bs_{args.data.split("/")[-2]}')
    writer = SummaryWriter(logdir='dump/asd')  # TODO: remove
    '''
    This is the main training loop. I left in this function the two main loops for the epochs and batches respectively.
    Since we implementing different training techniques, the actual body of the training is located in methods/ package,
    where we can more clearly organize the different training strategies, without having one single training body,
    riddled with flags and checks. The reference to the function we use for each method can be found in the 
    'train_method' structure above.
    '''
    # Training Epoch Loop
    for epoch in range(args.epochs):

        # Epoch progress bar
        tq = tqdm.tqdm(total=len(dl_train) * args.batch_size)
        tq.set_description(f'epoch {epoch + 1}')

        train_running_loss = []
        train_running_acc = []

        # Training Batch loop
        for i, (img, lbl) in enumerate(dl_train):
            '''
            All the important training stuff can be found following the train_ function. train_ function contains the
            value train_method corresponding to the args.method key. 
            '''
            loss, acc = train_(model, img, lbl, optimizer, criterion)
            # Update progress bar
            tq.update(args.batch_size)
            tq.set_postfix({'train_loss': '%.6f' % loss, 'train_acc': '%.6f' % acc})
            train_running_loss.append(loss)
            train_running_acc.append(acc)

            writer.add_scalar('training_loss', loss, epoch * args.batch_size + i)
            writer.add_scalar('training_acc', acc, epoch * args.batch_size + i)

        # Update learning rate
        lr_scheduler.step()

        # Close batch progress bar
        writer.add_scalar('epoch_loss', np.mean(train_running_loss), epoch)
        writer.add_scalar('epoch_acc', np.mean(train_running_loss), epoch)

        val_running_loss = []
        val_running_acc = []
        # Validation Batch Loop
        for i, (img, lbl) in enumerate(dl_val):
            loss, acc = val_(model, img, lbl, criterion)
            val_running_loss.append(loss)
            val_running_acc.append(acc)

        writer.add_scalar('val_loss', np.mean(val_running_loss), epoch)
        writer.add_scalar('val_acc', np.mean(val_running_acc), epoch)

        tq.set_postfix({'train_loss': '%.6f' % np.mean(train_running_loss),
                        'train_acc': '%.6f' % np.mean(train_running_acc),
                        'val_loss': '%.6f' % np.mean(val_running_loss),
                        'val_acc': '%.6f' % np.mean(val_running_acc)})
        tq.close()
    # Training concluded


if __name__ == '__main__':
    main()
