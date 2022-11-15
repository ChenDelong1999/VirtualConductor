import os
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from torch.utils.data import Dataset


class ConductorMotionDataset(Dataset):
    def __init__(self, sample_length, split, limit=None, root_dir='Dataset'):

        self.dataset_dir = os.path.join(root_dir, split)
        self.sample_length = sample_length
        self.name_list = os.listdir(self.dataset_dir)
        self.sample_idx = []
        self.dataset = dict()
        self.limit = limit

        accumlated_length = 0
        pbar = tqdm.tqdm(range(len(self.name_list)))
        for i in pbar:
            name = self.name_list[i]
            motion = np.load(os.path.join(self.dataset_dir, name, 'motion.npy'))
            mel = np.load(os.path.join(self.dataset_dir, name, 'mel.npy'))

            sample_num = int(motion.shape[0] / 30 / self.sample_length)
            pbar.set_description(f'Loading dataset: '
                                 f'{i + 1}/{len(self.name_list)} folder, '
                                 f'sample length: {int(motion.shape[0] / 30)} seconds, '
                                 f'split to {sample_num} samples')

            self.dataset[name] = {'motion': motion.astype(np.float32), 'mel': mel.astype(np.float32)}
            for j in range(sample_num):
                self.sample_idx.append([i, j * self.sample_length, (j + 1) * self.sample_length])

            accumlated_length += motion.shape[0] / 30
            if self.limit and accumlated_length / 3600 > self.limit:
                break

        print(f'Dataset initialized from {os.path.join(root_dir, split)}\n'
              f'\tdataset length:\t{round(len(self) * sample_length / 3600, 2)} hours\n'
              f'\tnum samples:\t{len(self)}\n'
              f'\tsample_length:\t{sample_length} seconds\n')

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, index):
        idx, start, end = self.sample_idx[index]
        name = self.name_list[idx]
        mel = self.dataset[name]['mel']
        motion = self.dataset[name]['motion']

        return mel[start * 90:end * 90, :], motion[start * 30:end * 30, :]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir')
    args = parser.parse_args()

    splits = ['test', 'val', 'train']

    for split in splits:
        dataset = ConductorMotionDataset(sample_length=60, split=split, limit=None, root_dir=args.dataset_dir)
        for i in range(len(dataset)):
            mel, motion = dataset[i]
            sample_info = f'{args.dataset_dir}/{split}/{i}'

            plt.figure(figsize=(12, 6))
            plt.subplot(211)
            plt.imshow(mel.T, aspect='auto', cmap='plasma')
            plt.colorbar()
            plt.title(f'Mel spectrogram (mel.npy) shape={mel.shape}')

            plt.subplot(212)
            plt.imshow(np.resize(motion, (1800, 26)).T, aspect='auto', cmap='plasma')
            plt.colorbar()
            plt.title(f'Conductor Motion (motion.npy) shape={motion.shape}')

            plt.subplots_adjust(hspace=0.3, top=0.85)
            plt.suptitle(sample_info)
            plt.show()

            kept_names = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
                          'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip']
            coordinate = ['x axis', 'y axis']
            plt.figure(figsize=(12, 6))
            for kept in range(13):
                for xy in range(2):
                    plt.subplot(13, 2, kept * 2 + xy + 1)
                    plt.plot(motion[:, kept, xy], linewidth=0.5, label=kept_names[kept] + ' - ' + coordinate[xy])
                    plt.legend(loc='upper right', fontsize=8)
                    plt.ylim(0, 1)
                    plt.xticks([])
                    plt.yticks([])
            plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.05)
            plt.suptitle(sample_info)
            plt.show()
