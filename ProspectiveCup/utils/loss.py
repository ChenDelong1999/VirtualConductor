import torch
import torch.nn as nn
import numpy as np
from scipy import signal

MSE = nn.MSELoss()


def strengh_contour_error(real_motion, fake_motion):
    real_v = torch.zeros_like(real_motion)
    fake_v = torch.zeros_like(fake_motion)
    real_v[:, 1:, :, :] = real_motion[:, :-1, :, :] - real_motion[:, 1:, :, :]
    fake_v[:, 1:, :, :] = fake_motion[:, :-1, :, :] - fake_motion[:, 1:, :, :]

    real_v = real_v.flatten(start_dim=2).mean(dim=2)
    fake_v = fake_v.flatten(start_dim=2).mean(dim=2)
    real_v = torch.abs(real_v)
    fake_v = torch.abs(fake_v)

    pool = torch.nn.AvgPool1d(kernel_size=60, stride=30)

    real_v_pool = pool(real_v.unsqueeze(0)).squeeze(0)
    fake_v_pool = pool(fake_v.unsqueeze(0)).squeeze(0)

    strengh_contour_error = MSE(fake_v_pool, real_v_pool)

    return torch.log(strengh_contour_error * 1e7 + 1)


def rhythm_density_error(real_motion_batch, fake_motion_batch):
    N, T, J, C = real_motion_batch.size()
    real_motion_batch = real_motion_batch.detach().cpu().numpy()
    fake_motion_batch = fake_motion_batch.detach().cpu().numpy()

    RDE_batch = np.zeros(N)
    for n in range(N):
        real_motion = real_motion_batch[n]
        fake_motion = fake_motion_batch[n]

        real_ryhthm_distribution = np.zeros([J, C, 16])
        fake_ryhthm_distribution = np.zeros([J, C, 16])
        for joint in range(J):
            for channel in range(C):
                f, t, S_real = signal.spectrogram(real_motion[:, joint, channel], 30)
                real_ryhthm_distribution[joint, channel, :] = S_real[1:17, :].mean(axis=1)
                f, t, S_fake = signal.spectrogram(fake_motion[:, joint, channel], 30)
                fake_ryhthm_distribution[joint, channel, :] = S_fake[1:17, :].mean(axis=1)

                f_psd, PSD_real_jc = signal.welch(real_motion[:, joint, channel], 30)
                f_psd, PSD_fake_jc = signal.welch(fake_motion[:, joint, channel], 30)

                if joint == channel == 0:
                    PSD_fake = PSD_fake_jc
                    PSD_real = PSD_real_jc
                else:
                    PSD_fake += PSD_fake_jc
                    PSD_real += PSD_real_jc

        bins = 26
        threshold = 6
        PSD_real /= 26
        PSD_fake /= 26
        RDE = ((PSD_real[threshold:bins] - PSD_fake[threshold:bins]) ** 2).mean()
        RDE_batch[n] = RDE

    return np.log(RDE_batch.mean() * 1e7 + 1)
