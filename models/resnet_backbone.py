import torch
from torchvision.models import resnet18, resnet34, resnet50
from torch.nn import Linear, Sequential
from utils.custom_exceptions import MethodNotSupportedError

resnets = {
    '18': resnet18,
    '34': resnet34,
    '50': resnet50,
}


def get_backbone(out_features, version='18', pretrained=False, weights=None):

    # Get the model from pytorch library
    try:
        net = resnets[version](pretrained=pretrained)
    except KeyError:
        raise MethodNotSupportedError(version)

    if weights:
        net.load_state_dict(torch.load(weights), strict=False)

    net.fc = Linear(in_features=net.fc.in_features, out_features=out_features, bias=True)

    return net
