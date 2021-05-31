from torchvision.models import resnet18, resnet34
from torch.nn import Linear
from utils.custom_exceptions import MethodNotSupportedError

resnets = {
    '18': resnet18,
    '34': resnet34,
}


def get_backbone(out_features, version='18'):

    # Get the model from pytorch library
    try:
        net = resnets[version](pretrained=False)
    except KeyError:
        raise MethodNotSupportedError(version)

    # Change classifier's output shape to match our training method
    net.fc = Linear(in_features=net.fc.in_features, out_features=out_features, bias=True)

    return net
