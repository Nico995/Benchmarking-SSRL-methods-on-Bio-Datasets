import argparse


def load_args():
    parser = argparse.ArgumentParser()

    # Training config
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--lr', type=float, default=0.025, help='learning rate used for train')
    parser.add_argument('--method', type=str, default='rotation', help='training method to use [rotation/jigsaw/...]')

    # Data
    parser.add_argument('--data', type=str, default='data/kather/', help='data path')
    parser.add_argument('--rescale_size', type=int, default=72, help='size for the rescaled image')
    parser.add_argument('--crop_size', type=int, default=64, help='size of the crops')

    # Environment resources config
    parser.add_argument('--threads', type=int, default=6, help='number of parallel threads for data loading')


    #
    args = parser.parse_args()

    return args