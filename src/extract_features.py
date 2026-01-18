import cv2
import os
import numpy as np

def extract(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    return [
        np.mean(L), np.mean(A), np.mean(B),
        np.var(L), np.var(A), np.var(B)
    ]

def load(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            X.append(extract(path))
            y.append(label)
    return np.array(X), np.array(y)
