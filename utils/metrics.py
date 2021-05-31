import torch


def accuracy(target, out):
    pred = torch.argmax(out.data, dim=1)
    correct = torch.sum(torch.eq(pred, target))

    return correct / target.shape[0]
