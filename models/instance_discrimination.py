import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from .resnet_backbone import get_backbone

import torch
from torch.autograd import Function
from torch import nn
import math


class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        t = params[0].item()
        batch_size = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(t)  # batchSize * N

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, grad_output):
        x, memory, y, params = self.saved_tensors
        batch_size = grad_output.size(0)
        t = params[0].item()
        momentum = params[1].item()

        # add temperature
        grad_output.data.div_(t)

        # gradient of linear
        grad_input = torch.mm(grad_output.data, memory)
        grad_input.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None


class LinearAverage(nn.Module):

    def __init__(self, code_size, n_instances, t=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        std = 1 / math.sqrt(code_size)
        self.nLem = n_instances

        self.register_buffer('params', torch.tensor([t, momentum]))
        std = 1. / math.sqrt(code_size / 3)
        self.register_buffer('memory', torch.rand(n_instances, code_size).mul_(2 * std).add_(-std))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class InstanceDiscrimination(nn.Module):
    def __init__(self, num_classes, version='18', weights=None):
        super(InstanceDiscrimination, self).__init__()
        self.backbone = get_backbone(out_features=num_classes[0], version=version, weights=weights)
        self.norm = Normalize(2)

        self.lemniscate = LinearAverage(num_classes[0], num_classes[1], 0.1, 0.5)

    def forward(self, x, y):
        x = self.backbone(x)
        x = self.norm(x)
        x = self.lemniscate(x, y)
        return x
