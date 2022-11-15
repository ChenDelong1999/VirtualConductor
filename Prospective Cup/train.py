import argparse
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn

from models.plain_model import Generator
from eval import Evaluator
from utils.dataset import ConductorMotionDataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def train(args):
    writer = SummaryWriter(comment='-baseline')
    args.logdir=writer.get_logdir()
    print(f'Logging to {args.logdir}')

    training_set = ConductorMotionDataset(
        sample_length=args.sample_length,
        split=args.training_set,
        limit=args.training_set_limit,
        root_dir=args.dataset_dir
        )
    train_loader = DataLoader(
        dataset=training_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True
        )
    evaluator = Evaluator(args)

    G = Generator().cuda()

    print('Start training')
    for epoch in range(args.epoch_num):
        pbar = tqdm.tqdm(enumerate(train_loader), total=train_loader.__len__())
        for step, (music, motion) in pbar:
            music = music.type(torch.FloatTensor).cuda()
            motion = motion.type(torch.FloatTensor).cuda()

            """
            add your training code
            """

            loss = 0

            writer.add_scalar('Training/Training Loss', loss, epoch * len(train_loader) + step)
            pbar.set_description(f'epoch: {epoch}, step: {step}, loss: {loss:.4f}')

        if epoch % args.evaluate_epoch == 0 or epoch == args.epoch_num:
            evaluator.evaluate(G, writer, epoch)


def main(args):
    options = vars(args)
    print('Args:')
    print('-' * 64)
    for key in options.keys():
        print(f'\t{key}:\t{options[key]}')
    print('-' * 64)

    train(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VirtualConductor-baseline')
    parser.add_argument('--dataset_dir', default='Dataset')
    parser.add_argument('--training_set', default='train')
    parser.add_argument('--training_set_limit', default=None, type=int, help='in: hours')
    parser.add_argument('--testing_set', default='val')
    parser.add_argument('--testing_set_limit', default=None, type=int, help='in: hours')

    parser.add_argument('--epoch_num', default=100, type=int, help='total epochs')
    parser.add_argument('--evaluate_epoch', default=5, type=int, help='interval between performing evaluation')

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--sample_length', default=30, type=int, help='in seconds')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

    args = parser.parse_args()

    main(args)
