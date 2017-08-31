from __future__ import absolute_import, division, print_function

from os import path
import numpy as np

N = 5
M = 10
BASE_DIR = "out"
DIR_FMT = "%02d"
FILE_FMT = "%02d.txt"

print("Loading data...")
exp_data = []
for i in range(N):
    exp_dir = path.join(BASE_DIR, DIR_FMT % i)
    probs = []
    labels = []

    for j in range(M):
        data = np.loadtxt(path.join(exp_dir, FILE_FMT % j), delimiter=", ")

        logits = np.zeros(data[:, :12].shape)
        lbls = np.zeros(data[:, 12].shape, dtype=np.int32)
        for k in range(data.shape[0]):
            idx = int(data[k, -1])
            logits[idx, :] = data[k, :12]
            lbls[idx] = data[k, 12]

        probs.append(np.exp(logits) / np.expand_dims(np.sum(np.exp(logits), 1), -1))
        labels.append(lbls)

    exp_data.append({'probs': probs, 'labels': labels})

print("Computing results")
pred = np.zeros(exp_data[0]['probs'][0].shape)
for i in range(N):
    for j in range(M):
        pred += exp_data[i]['probs'][j]

accuracy = np.sum(np.argmax(pred, 1) == exp_data[0]['labels'][0]) / pred.shape[0]
print("Accuracy:", accuracy)
