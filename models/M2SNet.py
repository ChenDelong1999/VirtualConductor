import torch
from torch import nn
from models.MusicEncoder import MusicEncoder
from models.MotionEncoder import MotionEncoder_STGCN


class M2SNet(nn.Module):

    def __init__(self):
        super(M2SNet, self).__init__()

        self.music_encoder = MusicEncoder()
        self.motion_encoder = MotionEncoder_STGCN()
        self.fuse_layer = nn.Sequential(
            nn.Conv1d(64+64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1), nn.Sigmoid(),
        )

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')

    def forward(self, x, y):
        hx = self.music_encoder(x)
        hy = self.motion_encoder(y)
        h_fuse = torch.cat([hx, hy], dim=2)
        out = self.fuse_layer(h_fuse.transpose(1, 2)).transpose(1, 2)
        return out

    def features(self, x, y):
        x_features = self.music_encoder.features(x)
        y_features = self.motion_encoder.features(y)
        return x_features, y_features