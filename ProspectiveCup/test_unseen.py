import os
import time
import numpy as np
import torch
from moviepy.editor import VideoFileClip, AudioFileClip
from models.Generator import Generator
import torch.utils.data as Data
from utils.music_utils import extract_mel_feature
import time
from utils.motion_utils import vis_motion


class TestDataset(Data.Dataset):
    def __init__(self, test_samles_dir):
        self.test_samles_dir = test_samles_dir
        self.name_list = os.listdir(test_samles_dir)
        if '.gitkeep' in self.name_list:
            self.name_list.remove('.gitkeep')
        print('found those test samples:', self.name_list)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        try:
            feature = np.load('test/cache/' + 'mel_' + name + '.npy')
            print('using cached features for', name)
        except FileNotFoundError:
            print('extracting Mel spectrogram for', name)
            end_time = time.time()
            feature = extract_mel_feature(self.test_samles_dir + name)
            print(f'Mel spectrogram extracted in {round(time.time() - end_time, 2)} seconds')
            np.save('test/cache/' + 'mel_' + name + '.npy', feature)

        return feature, name


def test(G, test_samles_dir='test\\test_samples\\', save_path='test/result'):
    G.eval()
    dataset = TestDataset(test_samles_dir=test_samles_dir)
    testloader = Data.DataLoader(dataset=dataset, batch_size=1)
    for step, (mel, name) in enumerate(testloader):
        name = name[0]
        print('evaluating {}/{} test sample:{}'.format(step+1, len(dataset), name))
        end_time = time.time()
        globle_limit = 60  # minute
        music_sr = 90
        motion = np.zeros([int(mel.size()[1] / (music_sr / 30)) + 1, 13, 2])
        for split in range(min(globle_limit, int(mel.size()[1] / 60 / music_sr) + 1)):
            if (split + 1) * 60 * music_sr <= mel.shape[1]:
                mel_step = mel[:, split * 60 * music_sr: (split + 1) * 60 * music_sr, :]
            else:
                end = mel.shape[1] - mel.shape[1] % music_sr
                mel_step = mel[:, split * 60 * music_sr: end, :]
                if end - split * 60 * music_sr < 5 * music_sr:
                    continue

            mel_step = mel_step.type(torch.FloatTensor).cuda()
            noise = torch.randn([1, int(mel_step.size()[1] / music_sr), 8])
            fake_step = G(mel_step, noise.cuda())
            fake_step = fake_step.detach().cpu().numpy()[0]
            motion[split * 60 * 30:split * 60 * 30 + fake_step.shape[0], :, :] = fake_step
        print(f'motion generated in {round(time.time()-end_time,2)} seconds')
        print('rendering video...')
        saved_video_file = vis_motion([motion], save_path=save_path, name=name)

        video = VideoFileClip(saved_video_file)
        video = video.set_audio((AudioFileClip(test_samles_dir + name)))
        video.write_videofile(saved_video_file + '.mp4')
        os.remove(saved_video_file)
        print('='*64)
    print('test finished')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()

    G = Generator().cuda()
    G.load_state_dict(torch.load(args.model))

    save_path = 'test/result/' + time.strftime("%Y-%m-%d_%H-%M-%S/", time.localtime())
    os.mkdir(save_path)

    test(G=G, save_path=save_path)
