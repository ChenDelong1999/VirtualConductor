import torch
from torch import nn
from models.MusicEncoder import MusicEncoder
from models.TCN import DialtedCNN


class PoseDecoderBiLSTM(nn.Module):
    """
    Args:
        feature_size (int): input feature dim

    Input: (batch_size, seq_len, feature_size)
    Output: (batch_size, seq_len, 20)

    """

    def __init__(self, input_szie, output_szie):
        super(PoseDecoderBiLSTM, self).__init__()

        self.LSTM = torch.nn.LSTM(input_size=input_szie, hidden_size=128, bidirectional=True, num_layers=2,
                                  batch_first=True, dropout=0.5)
        self.out = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, output_szie), nn.Sigmoid(),
                                 )

    def forward(self, input_feature):
        LSTM_out, _ = self.LSTM(input_feature)
        pose = self.out(LSTM_out)

        return pose


class PoseDecoderTCN(nn.Module):

    def __init__(self, input_szie, output_size):
        super(PoseDecoderTCN, self).__init__()
        self.TCN = DialtedCNN(input_size=input_szie, output_size=64, n_layers=6, n_channel=64, kernel_size=5)
        self.fc = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_size), nn.Sigmoid(),
        )

    def forward(self, input_feature):
        out = self.TCN(input_feature)
        out = self.fc(out)

        return out


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.music_encoder = MusicEncoder()
        self.tcn = PoseDecoderTCN(128, 26)
        # self.tcn = PoseDecoderBiLSTM(128, 26)
        self.noise_convTranspose = nn.Sequential(  # input: [N, 30, 8], output: [N, 900, 64], 30=2x3x5
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 16, kernel_size=11, stride=5, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(16, 32, kernel_size=5, stride=3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 64, kernel_size=6, stride=2, padding=2), nn.ReLU(),
        )
        self.noise_BN = nn.BatchNorm1d(64)

    def forward(self, x, noise):
        hx = self.music_encoder(x)
        hnoise = self.noise_convTranspose(noise.transpose(1, 2))
        hnoise = self.noise_BN(hnoise).transpose(1, 2)

        input = torch.cat([hx, hnoise], dim=2)
        y = self.tcn(input)
        N, L, C = y.size()
        y = y.view(N, L, 13, 2)

        return y

    def features(self, x, noise):
        hx = self.music_encoder(x)
        hnoise = self.noise_convTranspose(noise.transpose(1, 2))
        hnoise = self.noise_BN(hnoise).transpose(1, 2)

        input = torch.cat([hx, hnoise], dim=2)

        return input


class Generator_CVPR_LSTM(nn.Module):

    def __init__(self):
        super(Generator_CVPR_LSTM, self).__init__()
        self.lstm = PoseDecoderBiLSTM(20, 26)

    def forward(self, x, noise):
        y = self.lstm(x)
        N, L, C = y.size()
        y = y.view(N, L, 13, 2)

        return y

if __name__ == '__main__':
    LSTM = Generator_CVPR_LSTM()

    input = torch.zeros([3, 900, 20])
    print(LSTM(input, None).size())
