import os
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataset import ConductorMotionDataset
from utils.train_utils import PairBuilder, plot_hidden_feature


class M2SNet_evaluator():
    def __init__(self, args):
        self.batch_size = 2
        self.clip_length = args.clip_length
        self.sample_length = args.sample_length
        self.mode = args.mode
        self.save_path = 'checkpoints/M2SNet/'+self.mode
        if not os.path.isdir('checkpoints/M2SNet/'+self.mode):
            os.mkdir(self.save_path)
        self.testing_set = ConductorMotionDataset(sample_length=self.sample_length,
                                                  split=args.testing_set,
                                                  limit=args.testing_set_limit,
                                                  root_dir=args.dataset_dir)
        self.test_loader = DataLoader(dataset=self.testing_set, batch_size=self.batch_size, shuffle=True)
        self.pairBuilder = PairBuilder(args)

    def evaluate(self, M2SNet, writer, epoch, total_step):
        M2SNet.eval()

        print('| Evaluating M2SNet at Epoch {}'.format(epoch))

        accuracy_easy_all = []
        accuracy_hard_all = []
        accuracy_superhard_all = []

        sync_easy_all = []
        non_sync_easy_all = []
        sync_hard_all = []
        non_sync_hard_all = []
        sync_superhard_all = []
        non_sync_superhard_all = []

        for repet in range(1):
            pbar = tqdm.tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            for step, (music, motion) in pbar:
                if motion.shape[0] != self.batch_size:
                    continue
                music = music.type(torch.FloatTensor).cuda()
                motion = motion.type(torch.FloatTensor).cuda()

                # --- Easy Negatives --- #
                mel_1, mel_2, motion_1, motion_2 = self.pairBuilder.build_pairs(music, motion, 'easy')
                pred_easy_11 = M2SNet(mel_1, motion_1)
                pred_easy_12 = M2SNet(mel_1, motion_2)
                sync_easy_all.append(torch.mean(pred_easy_11).item())
                non_sync_easy_all.append(torch.mean(pred_easy_12).item())

                TP = np.sum(pred_easy_11.detach().cpu().numpy() > 0.5)
                TF = np.sum(pred_easy_12.detach().cpu().numpy() < 0.5)
                accuracy_easy = (TP + TF) / (self.batch_size * 2 * 30 * self.clip_length)
                accuracy_easy_all.append(accuracy_easy.item())

                # --- Hard Negatives --- #
                mel_1, mel_2, motion_1, motion_2 = self.pairBuilder.build_pairs(music, motion, 'hard')
                pred_hard_11 = M2SNet(mel_1, motion_1)
                pred_hard_12 = M2SNet(mel_1, motion_2)
                sync_hard_all.append(torch.mean(pred_hard_11).item())
                non_sync_hard_all.append(torch.mean(pred_hard_12).item())

                TP = np.sum(pred_hard_11.detach().cpu().numpy() > 0.5)
                TF = np.sum(pred_hard_12.detach().cpu().numpy() < 0.5)
                accuracy_hard = (TP + TF) / (self.batch_size * 2 * 30 * self.clip_length)
                accuracy_hard_all.append(accuracy_hard.item())

                # --- Super-hard Negatives --- #
                mel_1, mel_2, motion_1, motion_2 = self.pairBuilder.build_pairs(music, motion, 'super_hard')
                pred_superhard_11 = M2SNet(mel_1, motion_1)
                pred_superhard_12 = M2SNet(mel_1, motion_2)
                sync_superhard_all.append(torch.mean(pred_superhard_11).item())
                non_sync_superhard_all.append(torch.mean(pred_superhard_12).item())

                TP = np.sum(pred_superhard_11.detach().cpu().numpy() > 0.5)
                TF = np.sum(pred_superhard_12.detach().cpu().numpy() < 0.5)
                accuracy_superhard = (TP + TF) / (self.batch_size * 2 * 30 * self.clip_length)
                accuracy_superhard_all.append(accuracy_superhard.item())

        accuracy_easy_avg = np.mean(accuracy_easy_all)
        accuracy_hard_avg = np.mean(accuracy_hard_all)
        accuracy_superhard_avg = np.mean(accuracy_superhard_all)
        writer.add_scalars('M2SNet/accuracy', {'test_easy': accuracy_easy_avg}, total_step)
        writer.add_scalars('M2SNet/accuracy', {'test_hard': accuracy_hard_avg}, total_step)
        writer.add_scalars('M2SNet/accuracy', {'test_superhard': accuracy_superhard_avg}, total_step)
        writer.add_scalars('M2SNet/prediction_test', {'sync_easy': np.mean(sync_easy_all),
                                                      'non_sync_easy': np.mean(non_sync_easy_all)}, total_step)
        writer.add_scalars('M2SNet/prediction_test', {'sync_hard': np.mean(sync_hard_all),
                                                      'non_sync_hard': np.mean(non_sync_hard_all)}, total_step)
        writer.add_scalars('M2SNet/prediction_test', {'sync_superhard': np.mean(sync_superhard_all),
                                                      'non_sync_superhard': np.mean(non_sync_superhard_all)}, total_step)

        print('| Easy: %.5f | Hard: %.5f | Super-hard: %.5f' % (accuracy_easy_avg,accuracy_hard_avg,accuracy_superhard_avg))

        music_features, motion_features = M2SNet.features(mel_1.cuda(), motion_1.cuda())
        for i in range(len(music_features)):
            feature = plot_hidden_feature(music_features[i].transpose(1, 2))
            writer.add_image('M2SNet Music feature/layer {}'.format(i), feature, total_step, dataformats='HWC')
        for i in range(len(motion_features)):
            if i == 0 or i == len(motion_features) - 1 or i % 2 == 0:
                feature = plot_hidden_feature(motion_features[i].transpose(1, 2))
                writer.add_image('M2SNet Motion feature/layer {}'.format(i), feature, total_step, dataformats='HWC')

        torch.save(M2SNet.state_dict(), '{}/M2SNet_{}_{}.pt'.format(self.save_path, epoch, total_step))
        torch.save(M2SNet.state_dict(), '{}/M2SNet_last.pt'.format(self.save_path))

        M2SNet.train()
