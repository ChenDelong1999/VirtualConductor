import torch
from torch import nn


class Discriminator_1DCNN(nn.Module):
    def __init__(self):
        super(Discriminator_1DCNN, self).__init__()
        self.motion_encoder = nn.Sequential(
            nn.Conv1d(26, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),

            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.motion_encoder(x)
        x = self.fc(x.transpose(1, 2))
        x = torch.mean(x, dim=1)
        return x

    def features(self, x):
        features = []
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.motion_encoder(x)
        features.append(x)
        return features
