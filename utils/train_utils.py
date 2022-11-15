import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class PairBuilder:
    """
    Sampling Positive and Negative Pairs for M2S Learning
    music sampling rate: 90 Hz
    motion sampling rate: 30 Hz
    """

    def __init__(self, args):
        if args.clip_length > args.sample_length / 3:
            raise RuntimeError('part_length should less than sample_length/3')
        self.sample_length = args.sample_length
        self.clip_length = args.clip_length

    def build_pairs(self, music, motion, sampling_strategy):

        if sampling_strategy == 'easy':
            '''
            Easy negative pairs are select from different samples 
            (i.e. different piece of music) within a mini-batch. 
            '''
            start = np.random.rand() * (self.sample_length - self.clip_length)
            end = start + self.clip_length
            music_start, music_end = int(start * 90), int(end * 90)
            motion_start, motion_end = int(start * 30), int(end * 30)

            music_1 = music[:, music_start:music_end, :].type(torch.FloatTensor).cuda()
            motion_1 = motion[:, motion_start:motion_end, :].type(torch.FloatTensor).cuda()
            music_2 = music[:, music_start:music_end, :].type(torch.FloatTensor).cuda().flip(dims=[0])
            motion_2 = motion[:, motion_start:motion_end, :].type(torch.FloatTensor).cuda().flip(dims=[0])

            return music_1, music_2, motion_1, motion_2

        elif sampling_strategy == 'hard':
            '''
            Hard negative pairs are select from same samples. 
            We force the sampled to pairs have a range of at least 10 seconds.
            '''

            start_1 = np.random.rand() * (self.sample_length - self.clip_length - 10)
            start_2 = start_1 + 10 + np.random.rand() * (self.sample_length - self.clip_length - start_1 - 10)
            end_1 = start_1 + self.clip_length
            end_2 = start_2 + self.clip_length

            music_start_1, music_end_1 = int(start_1 * 90), int(end_1 * 90)
            motion_start_1, motion_end_1 = int(start_1 * 30), int(end_1 * 30)
            music_start_2, music_end_2 = int(start_2 * 90), int(end_2 * 90)
            motion_start_2, motion_end_2 = int(start_2 * 30), int(end_2 * 30)

            music_1 = music[:, music_start_1:music_end_1, :].type(torch.FloatTensor).cuda()
            motion_1 = motion[:, motion_start_1:motion_end_1, :].type(torch.FloatTensor).cuda()
            music_2 = music[:, music_start_2:music_end_2, :].type(torch.FloatTensor).cuda()
            motion_2 = motion[:, motion_start_2:motion_end_2, :].type(torch.FloatTensor).cuda()

            return music_1, music_2, motion_1, motion_2

        elif sampling_strategy == 'super_hard':
            '''
            Super-hard negative pairs are also select from same samples, 
            but they are sampled by random temporal shifts within the range of 0.5 second to 5 seconds
            '''
            start_1 = np.random.rand() * (self.sample_length - self.clip_length - 5)
            start_2 = np.random.rand() * (5 - 0.5) + start_1
            end_1 = start_1 + self.clip_length
            end_2 = start_2 + self.clip_length

            music_start_1, music_end_1 = int(start_1 * 90), int(end_1 * 90)
            motion_start_1, motion_end_1 = int(start_1 * 30), int(end_1 * 30)
            music_start_2, music_end_2 = int(start_2 * 90), int(end_2 * 90)
            motion_start_2, motion_end_2 = int(start_2 * 30), int(end_2 * 30)

            music_1 = music[:, music_start_1:music_end_1, :].type(torch.FloatTensor).cuda()
            motion_1 = motion[:, motion_start_1:motion_end_1, :].type(torch.FloatTensor).cuda()
            music_2 = music[:, music_start_2:music_end_2, :].type(torch.FloatTensor).cuda()
            motion_2 = motion[:, motion_start_2:motion_end_2, :].type(torch.FloatTensor).cuda()

            return music_1, music_2, motion_1, motion_2


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True


def plot_motion(fake_motion, motion):
    if type(fake_motion) != np.ndarray:
        fake_motion = fake_motion.cpu().detach().numpy()
    if type(motion) != np.ndarray:
        motion = motion.cpu().detach().numpy()
    kept_names = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
                          'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip']
    coordinate = ['x axis', 'y axis']

    fig = plt.figure(figsize=(18, 6))

    for kept in range(13):
        for xy in range(2):
            plt.subplot(13, 2, kept * 2 + xy + 1)
            plt.plot(motion[0, :, kept, xy], linewidth=0.5,
                     label=kept_names[kept] + ' - ' + coordinate[xy] + '- real', color='gray')
            plt.plot(fake_motion[0, :, kept, xy], linewidth=0.5,
                     label=kept_names[kept] + ' - ' + coordinate[xy] + '- fake', color='r')
            plt.legend(loc='upper right', fontsize=8)
            plt.ylim(0, 1)
            plt.xticks([])
            plt.yticks([])
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95, top=0.95, bottom=0.05)

    image_path = 'temp_.png'
    plt.savefig(image_path)
    image_PIL = Image.open(image_path)
    img = np.array(image_PIL)
    plt.close()
    time.sleep(0.5)
    return img


def plot_hidden_feature(hidden_feature):
    hidden_feature = hidden_feature.transpose(1, 2).cpu().detach().numpy()
    hidden_feature = hidden_feature.astype(np.float32)[0]
    plt.figure(figsize=(20, 2))
    plt.imshow(hidden_feature, cmap='plasma', aspect='auto')
    plt.colorbar()

    image_path = 'temp_hidden_feature.png'
    plt.savefig(image_path)
    image_PIL = Image.open(image_path)
    img = np.array(image_PIL)
    plt.close()
    time.sleep(0.1)
    return img
