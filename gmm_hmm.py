import numpy as np
import librosa
from scipy.fft import dct
from hmmlearn.hmm import GMMHMM
from data_loader import load_wavs


data_dir = "./recordings"  # Change only this path.
train, test = load_wavs(data_dir, sr=16000)

frames = np.vstack(list(train.values()))
mean = frames.mean(axis=0)
std = np.maximum(frames.std(axis=0), 1e-6)
# Normalize the MFCC features to zero mean and unit variance.
train = {name: (X - mean) / std for name, X in train.items()}
test = {name: (X - mean) / std for name, X in test.items()}

# Train ten independent GMM-HMMs, one for each digit.
models = {}
for digit in range(10):
    sequences = [X for name, X in train.items()
                 if int(name.split("_")[0]) == digit]
    if not sequences:
        raise ValueError(f"No training recordings for digit {digit}.")

    model = GMMHMM(
        n_components=3,  # 3 hidden states
        n_mix=3,         # 3 Gaussians per state
        n_iter=10,
        #init_params="mcw"
    )
    model.startprob_ = np.array([1.0, 0.0, 0.0])
    model.transmat_ = np.array([
        [0.8, 0.2, 0.0],
        [0.0, 0.8, 0.2],
        [0.0, 0.0, 1.0],
    ])
    X = np.vstack(sequences)
    lengths = [len(sequence) for sequence in sequences]
    model.fit(X, lengths=lengths)
    models[digit] = model
    print(f"Digit {digit}: training completed")

# Classify by Forward log-likelihood.
correct = 0
for name, X in test.items():
    scores = [models[digit].score(X) for digit in range(10)]
    prediction = int(np.argmax(scores))
    label = int(name.split("_")[0])
    correct += int(prediction == label)

print(f"Accuracy: {correct / len(test):.2%}")