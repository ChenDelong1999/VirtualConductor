import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from scipy import signal

MSE = nn.MSELoss()


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, motion_encoder, fake_motion, motion):
        real_feature = motion_encoder.features(motion)
        fake_feature = motion_encoder.features(fake_motion)
        loss_all = 0
        for i in range(len(real_feature)):
            loss_all += self.criterion(real_feature[i], fake_feature[i])

        return loss_all


class SyncLoss(nn.Module):
    def __init__(self, motion_encoder):
        super(SyncLoss, self).__init__()
        self.motion_encoder = motion_encoder.eval()
        self.criterion = nn.L1Loss()

    def forward(self, fake_motion, motion):
        real_feature = self.motion_encoder.features(motion)
        fake_feature = self.motion_encoder.features(fake_motion)
        loss_all = self.criterion(real_feature[-1], fake_feature[-1])
        '''loss_all = 0
        for i in range(len(self.weights)):
            fake = torch.max_pool1d(fake_feature[i], kernel_size=5, stride=3)
            real = torch.max_pool1d(real_feature[i], kernel_size=5, stride=3)
            loss_all += self.criterion(fake, real)'''

        return loss_all


def calc_gradient_penalty_ST(D, real_data, fake_data, term=None):
    if term is None:
        term = ['real', 'fake', 'real_fake', 'real_motion', 'fake_motion']
    loss = 0
    center = 0
    if 'real' in term:
        output = D(real_data.requires_grad_(True))
        gradients = autograd.grad(outputs=output, inputs=real_data, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_real = gradients.norm(2, dim=1)
        GP_real = ((norm_real - center) ** 2).mean()
        loss += GP_real

    if 'fake' in term:
        output = D(fake_data.requires_grad_(True))
        gradients = autograd.grad(outputs=output, inputs=fake_data, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_fake = gradients.norm(2, dim=1)
        GP_fake = ((norm_fake - center) ** 2).mean()
        loss += GP_fake

    if 'real_motion' in term:
        real_motion = real_data - real_data.mean(dim=1).unsqueeze(1)
        real_structure = real_data.mean(dim=1).unsqueeze(1)
        fake_structure = fake_data.mean(dim=1).unsqueeze(1)

        alpha = torch.rand(1).cuda()
        input = (alpha * real_motion + alpha * fake_structure + (1 - alpha) * real_structure).requires_grad_(True)
        output = D(input)
        gradients = autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_real_motion = gradients.norm(2, dim=1)
        GP_real_motion = ((norm_real_motion - center) ** 2).mean()
        loss += GP_real_motion

    if 'fake_motion' in term:
        fake_motion = fake_data - fake_data.mean(dim=1).unsqueeze(1)
        real_structure = real_data.mean(dim=1).unsqueeze(1)
        fake_structure = fake_data.mean(dim=1).unsqueeze(1)

        alpha = torch.rand(1).cuda()
        input = (alpha * fake_motion + alpha * fake_structure + (1 - alpha) * real_structure).requires_grad_(True)
        output = D(input)
        gradients = autograd.grad(outputs=output, inputs=input, grad_outputs=torch.ones(output.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_fake_motion = gradients.norm(2, dim=1)
        GP_fake_motion = ((norm_fake_motion - center) ** 2).mean()
        loss += GP_fake_motion

    if 'real_fake' in term:
        alpha = torch.Tensor(np.random.random((real_data.size(0), 1, 1, 1))).cuda()
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)
        disc_interpolates = D(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        norm_real_fake = gradients.norm()
        gradient_penalty = ((norm_real_fake - center) ** 2).mean()
        loss += gradient_penalty

    return loss


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

    '''real_v = real_v.detach().cpu().numpy()
        fake_v = fake_v.detach().cpu().numpy()
        real_v_pool = real_v_pool.detach().cpu().numpy()
        fake_v_pool = fake_v_pool.detach().cpu().numpy()'''

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
