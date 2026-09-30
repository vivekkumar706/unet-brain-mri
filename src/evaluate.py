"""Evaluate all checkpoints in weights/ on held-out test patients; save results table + overlays.
  python -m src.evaluate --data data/kaggle_3m
"""
import argparse, glob, json, os
import cv2, numpy as np, torch
from torch.utils.data import DataLoader
from .dataset import LGGDataset, list_pairs, patient_split
from .inference import load_model, overlay, predict_mask
from .losses_metrics import dice_score, iou_score
from .train import get_device


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args()
    device = get_device()
    pairs = list_pairs(a.data, patient_split(a.data)["test"])
    loader = DataLoader(LGGDataset(pairs), batch_size=16)
    os.makedirs("outputs/samples", exist_ok=True)
    results = {}
    for w in sorted(glob.glob("weights/*.pt")):
        name = os.path.basename(w)[:-3]
        if name == "best":
            continue
        model = load_model(w, device)
        d = i = 0.0
        n = fp = healthy_n = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                lg = model(x); b = x.size(0)
                d += dice_score(lg, y) * b; i += iou_score(lg, y) * b; n += b
                pred = (torch.sigmoid(lg) > 0.5).float()
                healthy = y.sum((1, 2, 3)) == 0
                healthy_n += healthy.sum().item()
                fp += (pred.sum((1, 2, 3))[healthy] > 0).sum().item()
        results[name] = {"dice": round(d / n, 4), "iou": round(i / n, 4),
                         "false_positive_rate_on_healthy_slices": round(fp / max(healthy_n, 1), 4),
                         "n_test_slices": n}
        print(name, results[name])
    json.dump(results, open("outputs/results.json", "w"), indent=2)

    if results:  # qualitative samples from the best model: image | green=truth | red=prediction
        best = max(results, key=lambda k: results[k]["dice"])
        model = load_model(f"weights/{best}.pt", device)
        tumor = [p for p in pairs if cv2.imread(p[1], 0).max() > 0]
        for k, idx in enumerate(np.random.RandomState(0).permutation(len(tumor))[:8]):
            ip, mp = tumor[idx]
            img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            gt = (cv2.imread(mp, 0) > 127).astype(np.uint8)
            pm, _ = predict_mask(model, img, device)
            row = np.hstack([img, overlay(img, gt, (0, 255, 0)), overlay(img, pm, (255, 0, 0))])
            cv2.imwrite(f"outputs/samples/sample_{k}.png", cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print("best:", best, "| samples saved to outputs/samples/")


if __name__ == "__main__":
    main()
