import time
import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import ConductorMotionDataset
from utils.loss import rhythm_density_error, strengh_contour_error
from PIL import Image
import matplotlib.pyplot as plt


class Evaluator():
    def __init__(self, args):
        self.batch_size = 1
        self.sample_length = args.sample_length
        self.save_path = os.path.join(args.logdir, 'checkpoints')
        os.mkdir(self.save_path)

        self.testing_set = ConductorMotionDataset(
            sample_length=args.sample_length,
            split=args.testing_set,
            limit=args.testing_set_limit,
            root_dir=args.dataset_dir
            )
        self.test_loader = DataLoader(
            dataset=self.testing_set, 
            batch_size=self.batch_size
            )
        self.MSE = nn.MSELoss()

    def evaluate(self, G, writer, epoch, save_checkpoints=True):
        print('Start evaluation at epoch {}'.format(epoch))
        G.eval()

        SD_fake_all = []
        SD_real_all = []
        MSE_all = []
        RDE_all = []
        SCE_all = []

        pbar = tqdm.tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        for step, (mel, real_motion) in pbar:
            if real_motion.shape[0] != self.batch_size:
                continue
            mel = mel.type(torch.FloatTensor).cuda()
            real_motion = real_motion.type(torch.FloatTensor).cuda()

            fake_motion = G(mel)

            fake_sd = torch.mean(torch.std(fake_motion, dim=1)).item()
            real_sd = torch.mean(torch.std(real_motion, dim=1)).item()
            SD_fake_all.append(fake_sd)
            SD_real_all.append(real_sd)

            mse = self.MSE(fake_motion, real_motion)
            MSE_all.append(mse.item())

            RDE = rhythm_density_error(real_motion, fake_motion)
            RDE_all.append(RDE)

            SCE = strengh_contour_error(real_motion, fake_motion)
            SCE_all.append(SCE.item())

        writer.add_scalars('Evaluation/Standard Deviation',{
            'generated': np.mean(SD_fake_all),
            'real': np.mean(SD_real_all)}, epoch)
        writer.add_scalar('Evaluation/Mean Squared Error (MSE)',  np.mean(MSE_all), epoch)
        writer.add_scalar('Evaluation/Rhythm Density Error (RDE)', np.mean(RDE_all), epoch)
        writer.add_scalar('Evaluation/Strengh Contour Error (SCE)', np.mean(SCE_all), epoch)

        print(
            f'MSE: {np.mean(MSE_all):.4f} | '
            f'RDE: {np.mean(RDE_all):.4f} | '
            f'SCE: {np.mean(SCE_all):.4f} | '
            f'SDP: {np.mean(SD_fake_all)/np.mean(SD_real_all)*100:.2f}%'
              )

        fig_motion = self.plot_motion(fake_motion, real_motion)
        writer.add_image("testing sample", fig_motion, epoch, dataformats='HWC')

        if save_checkpoints:
            torch.save(G.state_dict(), '{}/checkpoint_{}epoch.pt'.format(self.save_path, epoch))
            torch.save(G.state_dict(), '{}/checkpoint_latest.pt'.format(self.save_path))

        G.train()
        return

    def plot_motion(self, fake_motion, motion):
        if type(fake_motion) != np.ndarray:
            fake_motion = fake_motion.cpu().detach().numpy()
        if type(motion) != np.ndarray:
            motion = motion.cpu().detach().numpy()
        kept_names = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
                      'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip']
        coordinate = ['x axis', 'y axis']

        plt.figure(figsize=(16, 16))

        for kept in range(13):
            for xy in range(2):
                plt.subplot(13, 2, kept * 2 + xy + 1)
                plt.plot(motion[0, :, kept, xy], linewidth=1,
                         label=kept_names[kept] + ' - ' + coordinate[xy] + '- real', color='gray')
                plt.plot(fake_motion[0, :, kept, xy], linewidth=1,
                         label=kept_names[kept] + ' - ' + coordinate[xy] + '- fake', color='red')
                plt.legend(loc='upper right', fontsize=10)
                plt.ylim(0, 1)
                plt.xticks([])
                plt.yticks([])
        plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95, top=0.95, bottom=0.05)

        image_path = 'testing sample.png'
        plt.savefig(image_path)
        image_PIL = Image.open(image_path)
        img = np.array(image_PIL)
        plt.close()
        time.sleep(0.5)
        return img

