import argparse
import tqdm
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.backends.cudnn

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import models.M2SNet
from utils.dataset import ConductorMotionDataset
from M2SNet_eval import M2SNet_evaluator
from utils.train_utils import PairBuilder

torch.manual_seed(19990319)
torch.cuda.manual_seed(19990319)
np.random.seed(19990319)


def train(args):
    total_step = 0

    training_set = ConductorMotionDataset(sample_length=args.sample_length,
                                          split=args.training_set,
                                          limit=args.training_set_limit,
                                          root_dir=args.dataset_dir)
    train_loader = DataLoader(dataset=training_set, batch_size=args.batch_size, shuffle=False)

    M2SNet = models.M2SNet.M2SNet().cuda()
    M2SNet.init_weight()
    optimizer_M2S = torch.optim.Adam(M2SNet.parameters(), lr=0.001)

    evatuator = M2SNet_evaluator(args)
    pairBuilder = PairBuilder(args)
    writer = SummaryWriter(comment='_M2SNet_[{}]'.format(args.mode))

    ONE = torch.ones([args.batch_size, 1]).cuda()
    ZERO = torch.zeros([args.batch_size, 1]).cuda()
    BCE = nn.BCELoss()

    for epoch in range(args.num_epoch):
        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (music, motion) in pbar:
            if motion.shape[0] != args.batch_size:
                continue

            optimizer_M2S.zero_grad()
            if epoch == 0:
                # easy negatives are used for pre-training in the first epoch
                # since we find the models under hard or super-hard negatives are difficult to train from scratch.
                music_1, music_2, motion_1, motion_2 = pairBuilder.build_pairs(music, motion, sampling_strategy='easy')
            else:
                music_1, music_2, motion_1, motion_2 = pairBuilder.build_pairs(music, motion,
                                                                               sampling_strategy=args.sampling_mode)

            pred_11 = M2SNet(music_1, motion_1)
            pred_12 = M2SNet(music_1, motion_2)
            pred_22 = M2SNet(music_2, motion_2)
            pred_21 = M2SNet(music_2, motion_1)
            loss = BCE(pred_11.mean(dim=1), ONE) + BCE(pred_12.mean(dim=1), ZERO) + \
                   BCE(pred_22.mean(dim=1), ONE) + BCE(pred_21.mean(dim=1), ZERO)

            loss.backward()
            optimizer_M2S.step()

            ###############################################
            #                    Logging                  #
            ###############################################

            TP = np.sum(pred_11.detach().cpu().numpy() > 0.5)
            TF = np.sum(pred_12.detach().cpu().numpy() < 0.5)
            accuracy = (TP + TF) / (args.batch_size * args.clip_length * 2 * 30)

            writer.add_scalars('M2SNet/loss', {'train': loss.item()}, total_step)
            writer.add_scalars('M2SNet/accuracy', {'train': accuracy.item()}, total_step)
            writer.add_scalars('M2SNet/prediction_train', {'sync_train': torch.mean(pred_11).item(),
                                                           'non_sync_train': torch.mean(pred_12).item()}, total_step)
            pbar.set_description('Epoch: %d | step: %d | total step: %d | loss: %.5f | training accuracy %.5f'
                                 % (epoch, step, total_step, loss.item(), accuracy))
            total_step += 1
        torch.cuda.empty_cache()

        if epoch % args.evaluate_epoch == 0:
            evatuator.evaluate(M2SNet, writer, epoch, total_step)


def main(args):
    if args.mode == 'hard_test':
        args.sampling_mode = 'hard'
        args.training_set = 'test'
        args.testing_set = 'train'
        args.testing_set_limit = 5

    elif args.mode == 'easy':
        args.sampling_mode = 'easy'

    elif args.mode == 'hard':
        args.sampling_mode = 'hard'

    elif args.mode == 'super_hard':
        args.sampling_mode = 'super_hard'
    else:
        raise RuntimeError('Invalid args.mode!')

    print()
    print('=' * 64)
    print(f' - Starting Contrastive Learning Stage with [{args.mode}] Mode - ')
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
    parser = argparse.ArgumentParser(description='Contrastive Learning Stage')

    parser.add_argument('--mode', default='hard',
                        help='specify the training mode. '
                             '"easy": train with easy negatives (unstable). '
                             '"hard": train with hard negatives (best). '
                             '"super_hard": train with super-hard negatives. '
                             '"hard_test": train on test set for Mean Perceptual Error (MPE)')

    parser.add_argument('--dataset_dir', default='Dataset')
    parser.add_argument('--training_set', default='train')
    parser.add_argument('--training_set_limit', default=None)
    parser.add_argument('--testing_set', default='test')
    parser.add_argument('--testing_set_limit', default=None, help='using a subset of dataset')

    parser.add_argument('--num_epoch', default=400, help='total epochs')
    parser.add_argument('--evaluate_epoch', default=10, help='interval between evaluation')

    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--sample_length', default=30, help='sample length before random sampling (in second)')
    parser.add_argument('--clip_length', default=10, help='sampled pair length (in second)')

    args = parser.parse_args()

    main(args)
