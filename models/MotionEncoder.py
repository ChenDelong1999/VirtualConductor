import torch
import torch.nn as nn
from models.ST_GCN.ST_GCN import ST_GCN


class MotionEncoder_STGCN(nn.Module):
    def __init__(self):
        super(MotionEncoder_STGCN, self).__init__()
        self.graph_args = {}
        self.st_gcn = ST_GCN(in_channels=2,
                             out_channels=32,
                             graph_args=self.graph_args,
                             edge_importance_weighting=True,
                             mode='M2S')
        self.fc = nn.Sequential(nn.Conv1d(32 * 13, 64, kernel_size=1), nn.BatchNorm1d(64))

    def forward(self, input):
        input = input.transpose(1, 2)
        input = input.transpose(1, 3)
        input = input.unsqueeze(4)

        output = self.st_gcn(input)
        output = output.transpose(1, 2)
        output = torch.flatten(output, start_dim=2)
        output = self.fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def features(self, input):
        input = input.transpose(1, 2)
        input = input.transpose(1, 3)
        input = input.unsqueeze(4)

        output = self.st_gcn(input)
        output = output.transpose(1, 2)
        output = torch.flatten(output, start_dim=2)
        output = self.fc(output.transpose(1, 2)).transpose(1, 2)

        features = self.st_gcn.extract_feature(input)
        features.append(output.transpose(1, 2))

        return features


class MotionAutoEncoder(nn.Module):
    def __init__(self):
        super(MotionAutoEncoder, self).__init__()
        self.graph_args = {}
        self.encoder_stgcn = ST_GCN(in_channels=2,
                                    out_channels=16,
                                    graph_args=self.graph_args,
                                    edge_importance_weighting=True,
                                    mode='AE')
        self.encoder_fc = nn.Sequential(nn.Conv1d(16 * 13, 16, kernel_size=1), nn.BatchNorm1d(16))
        self.decoder = nn.Sequential(nn.Conv1d(16, 16, kernel_size=1),
                                     nn.BatchNorm1d(16), nn.ReLU(),
                                     nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.ConvTranspose1d(16, 16, stride=2, kernel_size=6, padding=2),
                                     nn.ReLU(),
                                     nn.ConvTranspose1d(16, 16, stride=3, kernel_size=5, padding=1),
                                     nn.ReLU(),
                                     nn.Conv1d(16, 26, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(),
                                     )

    def forward(self, input):
        N, T, _, _ = input.size()
        input = input.transpose(1, 2)
        input = input.transpose(1, 3)
        input = input.unsqueeze(4)

        hidden = self.encoder_stgcn(input)
        hidden = hidden.transpose(1, 2)
        hidden = torch.flatten(hidden, start_dim=2)
        hidden = self.encoder_fc(hidden.transpose(1, 2))

        reconstruction = self.decoder(hidden).transpose(1,2)
        reconstruction = reconstruction.view(N, T, 13, 2)

        return reconstruction, hidden.transpose(1,2)

