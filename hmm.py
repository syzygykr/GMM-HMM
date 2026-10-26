

import numpy as np
from scipy.special import logsumexp
from data_loader import load_wavs
import librosa
from scipy.fft import dct

def emission(X, mu, var):
    log_B = np.zeros((len(X), len(mu)))
    for j in range(len(mu)):
        log_B[:, j] = -0.5 * np.sum(
            np.log(2 * np.pi * var[j]) + (X - mu[j]) ** 2 / var[j], axis=1
        )
    return log_B

def forward(log_B, log_A):
    T, N = log_B.shape
    alpha = np.full((T, N), -np.inf)
    alpha[0, 0] = log_B[0, 0]  # Fixed pi = [1, 0, 0].
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = log_B[t, j] + logsumexp(
                alpha[t - 1] + log_A[:, j]
            )
    return alpha, logsumexp(alpha[-1])

def expectation(log_B, log_A):
    T, N = log_B.shape
    alpha, log_likelihood = forward(log_B, log_A)
    beta = np.zeros((T, N))  # log(1) = 0.

    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = logsumexp(log_A[i] + log_B[t + 1] + beta[t + 1])

    # gamma[t, j]: probability of state j at time t, given the recording.
    z = alpha + beta
    gamma = np.exp(z - logsumexp(z, axis=1, keepdims=True))

    # xi[t, i, j]: probability of transition i -> j at time t.
    xi = np.zeros((T - 1, N, N))
    for t in range(T - 1):
        z = alpha[t, :, None] + log_A + log_B[t + 1] + beta[t + 1]
        xi[t] = np.exp(z - logsumexp(z))
    return gamma, xi, log_likelihood


# Baum-Welch training for ONE digit. Each X is a separate recording.
def train_hmm(sequences, A, n_iter=10):
    # Initialize the states using the beginning, middle, and end of recordings 
    # because # of hidden state is 3.
    parts = [np.array_split(X, 3) for X in sequences]
    groups = [np.vstack([part[j] for part in parts]) for j in range(3)]
    mu = np.array([X.mean(axis=0) for X in groups])
    var = np.maximum([X.var(axis=0) for X in groups], 1e-3)

    for iteration in range(n_iter):
        with np.errstate(divide="ignore"): # log(0) = -inf
            log_A = np.log(A)  
        weight = np.zeros(3)
        x_sum = np.zeros_like(mu)
        x2_sum = np.zeros_like(mu)
        transitions = np.zeros((3, 3))
        total_ll = 0.0

        # E-step: process each recording separately, then accumulate statistics.
        for X in sequences:
            gamma, xi, ll = expectation(emission(X, mu, var), log_A)
            weight += gamma.sum(axis=0)
            x_sum += gamma.T @ X
            x2_sum += gamma.T @ (X ** 2)
            transitions += xi.sum(axis=0)
            total_ll += ll

        # M-step: update Gaussian means/variances and transition probabilities.
        for j in range(3):
            if weight[j] > 1e-8:
                mu[j] = x_sum[j] / weight[j]
                var[j] = np.maximum(x2_sum[j] / weight[j] - mu[j] ** 2, 1e-3)
            if transitions[j].sum() > 1e-8:
                A[j] = transitions[j] / transitions[j].sum()
        # This likelihood was evaluated BEFORE the current M-step.
        print(f"  E-step {iteration + 1:2d}: log-likelihood = {total_ll:.2f}")
    return A, mu, var

def extract_mfcc(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=16000, n_fft=400, win_length=400, hop_length=160,
        n_mels=40, fmax=4000, norm=None, power=2.0,
    )
    log_mel = np.log(mel + 1e-10)
    return dct(log_mel, type=2, axis=0, norm="ortho")[:13].T.astype(float)


if __name__ == "__main__":
    data_dir = "./recordings"  # Change only this path.
    train, test = load_wavs(data_dir, sr=16000)

    frames = np.vstack(list(train.values()))
    mean = frames.mean(axis=0)
    std = np.maximum(frames.std(axis=0), 1e-6)
    # Normalize the MFCC features to zero mean and unit variance.
    train = {name: (X - mean) / std for name, X in train.items()}
    test = {name: (X - mean) / std for name, X in test.items()}

    A_init = np.array([[0.8, 0.2, 0.0],
                  [0.0, 0.8, 0.2],
                  [0.0, 0.0, 1.0]])
    # Train ten independent HMMs, one for each digit.
    models = {}
    for digit in range(10):
        sequences = [X for name, X in train.items()
                     if int(name.split("_")[0]) == digit]
        print(f"Training digit {digit}")
        models[digit] = train_hmm(sequences, A_init)

    # Classify by Forward log-likelihood.
    correct = 0
    for name, X in test.items():
        scores = []
        for digit in range(10):
            A, mu, var = models[digit]
            with np.errstate(divide="ignore"):
                log_A = np.log(A)
            _, score = forward(emission(X, mu, var), log_A)
            scores.append(score)
        prediction = int(np.argmax(scores))
        correct += int(prediction == int(name.split("_")[0]))
    print(f"Accuracy: {correct / len(test):.2%}")
