import torch
from torch import nn


class Conv2dResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, residual=True):
        super(Conv2dResLayer, self).__init__()
        self.conv2d_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    padding_mode='reflect'),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU())
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.conv2d_layer(x)
        res = self.residual(x)
        return out + res


class MusicEncoder(nn.Module):
    def __init__(self):
        super(MusicEncoder, self).__init__()

        self.conv1 = nn.Sequential(Conv2dResLayer(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), residual=False),
                                   Conv2dResLayer(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   Conv2dResLayer(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 2), padding=(2, 2)))
        self.conv2 = nn.Sequential(Conv2dResLayer(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   Conv2dResLayer(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.MaxPool2d(kernel_size=(5, 5), stride=(3, 2), padding=(2, 2)))
        self.conv3 = nn.Sequential(Conv2dResLayer(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   Conv2dResLayer(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)))
        self.conv4 = nn.Sequential(nn.Conv1d(32 * 16, 64, kernel_size=1, stride=1),nn.BatchNorm1d(64))

    def forward(self, x):
        mel = x.unsqueeze(1)
        h1 = self.conv1(mel)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h3 = h3.transpose(1, 2).flatten(start_dim=2).transpose(1, 2)
        h4 = self.conv4(h3).transpose(1, 2)
        return h4

    def features(self, x):
        mel = x.unsqueeze(1)
        h1 = self.conv1(mel)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h3 = h3.transpose(1, 2).flatten(start_dim=2).transpose(1, 2)
        h4 = self.conv4(h3).transpose(1, 2)

        h1 = torch.transpose(h1, 1, 2)
        h1 = torch.flatten(h1, 2)
        h2 = torch.transpose(h2, 1, 2)
        h2 = torch.flatten(h2, 2)
        h3 = torch.transpose(h3, 1, 2)
        h3 = torch.flatten(h3, 2)

        return [x.transpose(1, 2), h1.transpose(1, 2), h2.transpose(1, 2), h3.transpose(1, 2), h4.transpose(1, 2)]
