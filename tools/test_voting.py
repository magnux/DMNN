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

for i in range(N):
    print('Model', i)

    loss_mean = 0
    acc_mean = 0
    count = 0
    for P, L in zip(exp_data[i]['probs'], exp_data[i]['labels']):
        for p, l in zip(P, L):
            loss_mean += -np.log(p[l])
            acc_mean += int(np.argmax(p) == l)
            count += 1
    print("Mean loss:", loss_mean / count, "Mean accuracy:", acc_mean / count)

    P = sum(exp_data[i]['probs']) / M
    L = exp_data[i]['labels'][0]
    loss_mean = 0
    acc_mean = 0
    count = 0
    for p, l in zip(P, L):
        loss_mean += -np.log(p[l])
        acc_mean += int(np.argmax(p) == l)
        count += 1
    print("Loss of mean:", loss_mean / count, "Accuracy of mean:", acc_mean / count)
