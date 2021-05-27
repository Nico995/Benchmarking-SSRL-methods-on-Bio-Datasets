from torchvision.models import resnet18
from torch.nn import Linear


def get_model(out_features):
    # Get the model from pytorch library
    net = resnet18(pretrained=False)
    # Change classifier's output shape to match our training method
    net.fc = Linear(in_features=net.fc.in_features, out_features=out_features, bias=True)

    return net
