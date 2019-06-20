import torch


def one_hot(index, vocab_size=1000):
    output = torch.zeros(index.size(0), vocab_size).to(index.device)
    output.scatter_(1, index.unsqueeze(-1), 1)
    return output


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1) * 255
