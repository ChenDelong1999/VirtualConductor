import time
import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models.M2SNet
from models.Discriminator import Discriminator_1DCNN
from models.Generator import Generator
from utils.dataset import ConductorMotionDataset
from utils.loss import SyncLoss, rhythm_density_error, strengh_contour_error
from utils.train_utils import plot_motion


class M2SGAN_Evaluator():
    def __init__(self, args):
        self.batch_size = 1
        self.sample_length = args.sample_length
        self.mode = args.mode
        self.save_path = 'checkpoints/M2SGAN/' + self.mode + time.strftime("_%a-%b-%d_%H-%M-%S", time.localtime())
        os.mkdir(self.save_path)

        self.testing_set = ConductorMotionDataset(sample_length=args.sample_length,
                                                  split=args.testing_set,
                                                  limit=args.testing_set_limit,
                                                  root_dir=args.dataset_dir)
        self.test_loader = DataLoader(dataset=self.testing_set, batch_size=self.batch_size, shuffle=True)
        print('testing set initialized, {} samples, {} hours'
              .format(len(self.testing_set), round(len(self.testing_set) * args.sample_length / 3600, 2)))

        self.MSE = nn.MSELoss()

        M2SNet = models.M2SNet.M2SNet().cuda()
        M2SNet.load_state_dict(torch.load(args.M2SNet))
        M2SNet.eval()
        self.perceptual_loss = SyncLoss(M2SNet.motion_encoder)

    def evaluate(self, G, D, perceptual_loss, writer, epoch, total_step, save_checkpoints=True):
        G.eval()
        D.eval()

        print('| Evaluating M2SGAN at Epoch {}'.format(epoch))

        # Realism
        SD_fake_all = []
        SD_real_all = []
        W_dis_all = []

        # Consistency
        MSE_all = []
        MPE_all = []
        loss_sync_all = []
        RDE_all = []
        SCE_all = []

        pbar = tqdm.tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for step, (mel, real_motion) in pbar:
            if real_motion.shape[0] != self.batch_size:
                continue
            mel = mel.type(torch.FloatTensor).cuda()
            real_motion = real_motion.type(torch.FloatTensor).cuda()

            noise = torch.randn([self.batch_size, self.sample_length, 8]).cuda()
            fake_motion = G(mel, noise)

            # ----------- #
            #   Realism   #
            # ----------- #

            # Standard Deviation
            fake_sd = torch.mean(torch.std(fake_motion, dim=1)).item()
            real_sd = torch.mean(torch.std(real_motion, dim=1)).item()
            SD_fake_all.append(fake_sd)
            SD_real_all.append(real_sd)

            # W Distance
            real_output_D = D(real_motion)
            fake_output_D = D(fake_motion.detach())
            W_dis_all.append((real_output_D - fake_output_D).detach().cpu().numpy().mean())

            # ----------- #
            # Consistency #
            # ----------- #

            # Mean Squared Error
            mse = self.MSE(fake_motion, real_motion)
            MSE_all.append(mse.item())

            # Mean Perceptual Error
            mpe = self.perceptual_loss(fake_motion, real_motion)
            MPE_all.append(mpe.item())

            # Perceptual Loss
            loss_sync = perceptual_loss(fake_motion, real_motion)
            loss_sync_all.append(loss_sync.item())

            # Rhythm Density Error
            RDE = rhythm_density_error(real_motion, fake_motion)
            RDE_all.append(RDE)

            # Strengh Contur Error
            SCE = strengh_contour_error(real_motion, fake_motion)
            SCE_all.append(SCE.item())

        writer.add_scalars('M2SGAN_Realism/W_distance',
                           {'test': np.array(W_dis_all).mean()}, total_step)
        writer.add_scalars('M2SGAN_Realism/Standard Deviation',
                           {'test': np.mean(SD_fake_all),
                            'real': np.mean(SD_real_all)}, total_step)

        writer.add_scalars('M2SGAN_Consistency/MSE Loss',
                           {'test': np.mean(MSE_all)}, total_step)
        writer.add_scalars('M2SGAN_Consistency/Sync Loss',
                           {'test': np.mean(loss_sync_all)}, total_step)

        writer.add_scalars('M2SGAN_Consistency/Sync Error (SE)', {'test': np.mean(MPE_all)}, total_step)
        writer.add_scalars('M2SGAN_Consistency/Rhythm Density Error (RDE)', {'test': np.mean(RDE_all)}, total_step)
        writer.add_scalars('M2SGAN_Consistency/Strengh Contour Error (SCE)', {'test': np.mean(SCE_all)}, total_step)

        print('| MPE: %.5f | RDE: %.5f | SCE: %.5f' % (np.mean(MPE_all), np.mean(RDE_all), np.mean(SCE_all)))

        fig_motion = plot_motion(fake_motion, real_motion)
        writer.add_image("M2SGAN training sample", fig_motion, total_step, dataformats='HWC')

        if save_checkpoints:
            torch.save(G.state_dict(), '{}/M2SGAN_Generator_{}_{}.pt'.format(self.save_path, epoch, total_step))
            torch.save(G.state_dict(), '{}/M2SGAN_Generator_last.pt'.format(self.save_path))
            torch.save(D.state_dict(), '{}/M2SGAN_Discriminator_{}_{}.pt'.format(self.save_path, epoch, total_step))
            torch.save(D.state_dict(), '{}/M2SGAN_Discriminator_last.pt'.format(self.save_path))

        G.train()
        D.train()

        return


if __name__ == '__main__':

    M2SNet = models.M2SNet.M2SNet().cuda()
    M2SNet.load_state_dict(torch.load('checkpoints/M2SNet/_M2SNet_hard_Latest.pt'))
    M2SNet.eval()
    perceptual_loss = SyncLoss(M2SNet.motion_encoder)
    evaluator = M2SGAN_Evaluator()

    exps = os.listdir('checkpoints/M2SGAN/Ablation')
    for exp in exps:
        if exp == 'CNN-LSTM':
            continue
        writer = SummaryWriter(comment='_post_eval_' + exp)
        G_ckpts = os.listdir('checkpoints/M2SGAN/Ablation/{}/Generator'.format(exp))
        G_ckpts.sort(key=lambda x: int(x.split('_')[-2]))
        D_ckpts = os.listdir('checkpoints/M2SGAN/Ablation/{}/Discriminator'.format(exp))
        D_ckpts.sort(key=lambda x: int(x.split('_')[-2]))
        print(exp, G_ckpts)
        for i in range(len(G_ckpts)):
            if i > len(G_ckpts) - 10:
                epoch = int(G_ckpts[i].split('_')[-2])
                global_step = int(G_ckpts[i].split('_')[-1].split('.')[0])
                G = Generator().cuda()
                G.load_state_dict(torch.load('checkpoints/M2SGAN/{}/Generator/{}'.format(exp, G_ckpts[i])))
                D = Discriminator_1DCNN().cuda()
                D.load_state_dict(torch.load('checkpoints/M2SGAN/{}/Discriminator/{}'.format(exp, D_ckpts[i])))

                evaluator.evaluate(G, D, perceptual_loss, writer, epoch, global_step, save_checkpoints=False)
