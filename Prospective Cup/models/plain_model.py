import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, x):
        """
        input:
            torch.tensor, size=(batch_size, 90*args.sample_length, 128)
        output:
            torch.tensor, size=(batch_size, 30*args.sample_length, 13, 2)
        """
        batch_size, time_steps, _ = x.size()
        y = torch.zeros([batch_size, int(time_steps/3), 13, 2]).cuda()

        return y

