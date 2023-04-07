import argparse
import tqdm
import numpy as np
import matplotlib

# matplotlib.use('TkAgg')

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import models.M2SNet
from models.Generator import Generator
from models.Discriminator import Discriminator_1DCNN
from M2SGAN_eval import M2SGAN_Evaluator
from utils.dataset import ConductorMotionDataset
from utils.train_utils import freeze, unfreeze
from utils.loss import calc_gradient_penalty_ST, SyncLoss, rhythm_density_error, strengh_contour_error, \
    FeatureMatchingLoss

torch.manual_seed(19990319)
torch.cuda.manual_seed(19990319)
np.random.seed(19990319)


def train(args):
    training_set = ConductorMotionDataset(sample_length=args.sample_length,
                                          split=args.training_set,
                                          limit=args.training_set_limit,
                                          root_dir=args.dataset_dir)
    train_loader = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    M2SNet = models.M2SNet.M2SNet().cuda()
    M2SNet.load_state_dict(torch.load(args.M2SNet))
    M2SNet.eval()
    perceptual_loss = SyncLoss(M2SNet.motion_encoder)

    MSE = nn.MSELoss()

    G = Generator().cuda()
    if args.transfer_music_encoder:
        G.music_encoder.load_state_dict(M2SNet.music_encoder.state_dict())
    if not args.train_music_encoder:
        freeze(G.music_encoder)
    optimizer_G = torch.optim.RMSprop(G.parameters(), lr=args.lr)

    D = Discriminator_1DCNN().cuda()

    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=args.lr)
    writer = SummaryWriter(comment='_M2SGAN_[{}]'.format(args.mode))
    evaluator = M2SGAN_Evaluator(args)

    total_step = 0

    SD_real_all = []
    for epoch in range(args.epoch_num):
        pbar = tqdm.tqdm(enumerate(train_loader), total=train_loader.__len__())
        for step, (music, real_motion) in pbar:
            if real_motion.shape[0] != args.batch_size:
                continue
            music = music.type(torch.FloatTensor).cuda()
            real_motion = real_motion.type(torch.FloatTensor).cuda()
            optimizer_G.zero_grad()

            noise = torch.randn([args.batch_size, args.sample_length, 8]).cuda()
            fake_motion = G(music, noise)

            # ------------------------ #
            #    train Discriminator   #
            # ------------------------ #
            for critic_i in range(args.CRITIC_ITERS):
                optimizer_D.zero_grad()
                real_output_D = D(real_motion)
                fake_output_D = D(fake_motion.detach())

                Loss_D_real = -torch.mean(real_output_D)
                Loss_D_fake = torch.mean(fake_output_D)

                gradient_penalty_Dr = calc_gradient_penalty_ST(D, real_motion.data, fake_motion.data,
                                                               term=['real_fake'])
                Loss_D = Loss_D_real + Loss_D_fake + args.w_gp * gradient_penalty_Dr
                Loss_D.backward()
                optimizer_D.step()

            # ----------------------- #
            #     train Generator     #
            # ----------------------- #
            optimizer_G.zero_grad()
            mse_loss = MSE(fake_motion, real_motion)
            Loss_adv = -torch.mean(D(fake_motion))
            sync_loss = perceptual_loss(fake_motion, real_motion)

            Loss_G = args.w_mse * mse_loss + args.w_adv * Loss_adv + args.w_sync * sync_loss
            Loss_G.backward()
            optimizer_G.step()

            ###############################################
            #                    Logging                  #
            ###############################################

            W_dis = torch.mean(real_output_D).item() - torch.mean(fake_output_D).item()
            writer.add_scalars('M2SGAN_Realism/W_distance', {'train': W_dis}, total_step)
            writer.add_scalars('M2SGAN_Realism/Standard Deviation',
                               {'train': torch.mean(torch.std(fake_motion, dim=1))}, total_step)
            SD_real_all.append(torch.mean(torch.std(real_motion, dim=1)).item())

            writer.add_scalars('M2SGAN_Consistency/MSE Loss',
                               {'train': mse_loss.item()}, total_step)
            writer.add_scalars('M2SGAN_Consistency/Perceptual Loss',
                               {'train': sync_loss.item()}, total_step)

            writer.add_scalars('M2SGAN_Consistency/Rhythm Density Error (RDE)',
                               {'train': rhythm_density_error(real_motion, fake_motion)}, total_step)
            writer.add_scalars('M2SGAN_Consistency/Strengh Contour Error (SCE)',
                               {'train': strengh_contour_error(real_motion, fake_motion)}, total_step)

            pbar.set_description('Epoch: %d | step: %d | total step: %d '
                                 '| MSE: %.5f | sync loss: %.5f | Wasserstein distance: %.5f'
                                 % (epoch, step, total_step,
                                    mse_loss.item(), sync_loss.item(), W_dis))
            total_step += 1
        torch.cuda.empty_cache()
        if epoch % args.evaluate_epoch == 0 or epoch == 0 or epoch == args.epoch_num:
            evaluator.evaluate(G, D, perceptual_loss, writer, epoch, total_step)
            writer.add_scalars('M2SGAN_Realism/Standard Deviation',
                               {'train_real': np.mean(SD_real_all)}, total_step)


def main(args):
    print()
    print('=' * 64)
    print(f' - Starting Generative Learning Stage - ')
    print('=' * 64)
    print()

    options = vars(args)
    print('Args:')
    print('-' * 64)
    for key in options.keys():
        print(f'\t{key}:\t{options[key]}')
    print('-' * 64)
    print()

    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Learning Stage')
    parser.add_argument('--mode', default='hard',
                        help='specify the training mode. '
                             '"easy": train with easy negatives (unstable). '
                             '"hard": train with hard negatives (best). '
                             '"super_hard": train with super-hard negatives. '
                             '"hard_test": train on test set for Mean Perceptual Error (MPE)')
    parser.add_argument('--M2SNet', default='checkpoints/M2SNet/hard/M2SNet_last.pt')
    parser.add_argument('--transfer_music_encoder', default=True)
    parser.add_argument('--train_music_encoder', default=False)
    parser.add_argument('--M2SNet_test', default='checkpoints/M2SNet/hard_test/M2SNet_last.pt',
                        help='to calculate sync error')

    parser.add_argument('--dataset_dir', default='Dataset')
    parser.add_argument('--training_set', default='train')
    parser.add_argument('--training_set_limit', default=None, help='in: hours')
    parser.add_argument('--testing_set', default='test')
    parser.add_argument('--testing_set_limit', default=None, help='in: hours')

    parser.add_argument('--epoch_num', default=200, help='total epochs')
    parser.add_argument('--evaluate_epoch', default=10, help='interval between performing evaluation')

    parser.add_argument('--batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--sample_length', default=30, help='in: seconds')
    parser.add_argument('--CRITIC_ITERS', default=5)
    parser.add_argument('--lr', default=0.0005, help='learning rate')

    parser.add_argument('--w_adv', default=1, help='weight for adversarial loss')
    parser.add_argument('--w_sync', default=0.05, help='weight for sync loss')
    parser.add_argument('--w_mse', default=0, help='weight for MSE loss')
    parser.add_argument('--w_gp', default=10, help='weight for gradient penalty')

    args = parser.parse_args()

    main(args)
