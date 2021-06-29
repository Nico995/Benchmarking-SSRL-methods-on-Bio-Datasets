import torch


def accuracy(out, target):
    pred = torch.argmax(out, dim=1)
    correct = torch.sum(torch.eq(pred, target))

    return correct / target.shape[0]


def corrects(out, target):
    pred = torch.argmax(out, dim=1)
    correct = torch.sum(torch.eq(pred, target))

    return correct
