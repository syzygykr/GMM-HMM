from pathlib import Path

import librosa
from scipy.fft import dct
import numpy as np


def load_wavs(data_dir, sr=16000):
    train_data = {}
    test_data = {}

    for path in sorted(Path(data_dir).glob("*.wav")):
        # filename: label_speaker_index.wav
        digit, speaker, index = path.stem.split("_")

        x, _ = librosa.load(path, sr=sr, mono=True)

        # PreProcessing (MFCC)
        y = extract_mfcc(x)

        if int(index) < 5:
            test_data[path.stem] = y
        else:
            train_data[path.stem] = y

    return train_data, test_data

def extract_mfcc(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=16000, n_fft=400, win_length=400, hop_length=160,
        n_mels=40, fmax=4000, norm=None, power=2.0,
    )
    log_mel = np.log(mel + 1e-10)
    return dct(log_mel, type=2, axis=0, norm="ortho")[:13].T.astype(float)
