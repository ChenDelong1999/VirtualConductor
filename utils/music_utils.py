import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import cv2


def extract_mel_feature(audio_file, mel_len_90fps=None):
    y, sr = librosa.load(audio_file)
    if mel_len_90fps is None:
        mel_len_90fps = int(len(y) / sr * 90)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=256)
    mel_dB = librosa.power_to_db(mel, ref=np.max)

    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(mel_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    # plt.show()

    norm_mel = np.flip(np.abs(mel_dB + 80) / 80, 0)
    resized_mel = cv2.resize(norm_mel, (mel_len_90fps, norm_mel.shape[0]))
    return resized_mel.T
