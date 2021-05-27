import numpy as np

classes = ["Tumor", "Stroma", "Complex", "Lympho", "Debris", "Mucosa", "Adipose", "Empty"]


def class_idx_to_name(idx):
    return classes[idx]


def batch_to_plottable_image(batch):
    return np.moveaxis(np.array(batch[0].detach().cpu()), 0, -1)


def dataset_name(data_path):
    """
    This function returns the dataset name from the peth. For convention, we will consider the name of the data root
     folder as the dataset name.
    Args:
        data_path (str): Filesystem path to reach the dataset root folder

    Returns:
        str: Dataset name
    """
    return list(filter(bool, data_path.split('/')))[-1]
