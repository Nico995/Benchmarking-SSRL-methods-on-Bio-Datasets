from torch.utils.data import DataLoader

from parser import load_args
from utils import dataset_name, config as cf
from utils.custom_exceptions import *

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
        # ds_val = cf.dataset[dataset_name(args.data)](args, mode='val', ext=ext, get_indices=False)
    except KeyError:
        raise DatasetNotSupportedError(dataset_name(args.data))

    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, num_workers=args.threads, drop_last=True,
                          shuffle=True)

    # dl_val = DataLoader(ds_val, batch_size=args.val_batch_size, num_workers=args.threads, drop_last=True, shuffle=True)
    for epoch in range(100):
        for i, data in enumerate(dl_train):
            image, label = data
            print(image.shape)


if __name__ == "__main__":
    main()
