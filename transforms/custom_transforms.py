import torchvision.transforms.functional as TF
import random

# The network needs to predict a class instead of a rotation.
# This data structure holds the indexing of such correspondence.
rotation_labels = {
    0: 0,
    90: 1,
    180: 2,
    270: 3
}


class DiscreteRandomRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        """
        Args:
            angles ([int]): list of angles from which the rotation method can choose from.
        """
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle), rotation_labels[angle]
