"""Kaggle LGG MRI segmentation dataset (kaggle_3m/<patient>/*.tif + *_mask.tif).
Splits are PATIENT-level to avoid leakage between slices of the same patient."""
import glob, os, random
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
IMG_SIZE = 256


def patient_split(root, seed=42, val_frac=0.15, test_frac=0.15):
    patients = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    random.Random(seed).shuffle(patients)
    n = len(patients)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    return {"test": patients[:n_test], "val": patients[n_test:n_test + n_val], "train": patients[n_test + n_val:]}


def list_pairs(root, patients):
    pairs = []
    for p in patients:
        for m in glob.glob(os.path.join(root, p, "*_mask.tif")):
            pairs.append((m.replace("_mask", ""), m))
    return sorted(pairs)


def get_transforms(train):
    norm = [A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=MEAN, std=STD)]
    if not train:
        return A.Compose(norm)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        *norm,
    ])


class LGGDataset(Dataset):
    def __init__(self, pairs, train=False):
        self.pairs, self.tf = pairs, get_transforms(train)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        out = self.tf(image=img, mask=mask)
        return torch.from_numpy(out["image"].transpose(2, 0, 1)).float(), torch.from_numpy(out["mask"])[None].float()
