"""Train a model. Examples:
  python -m src.train --data data/kaggle_3m --arch unet --epochs 40
  python -m src.train --data data/kaggle_3m --arch resnet34 --pretrained --epochs 25
"""
import argparse, json, os, time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import LGGDataset, list_pairs, patient_split
from .losses_metrics import DiceBCELoss, dice_score, iou_score
from .model import build_model


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(model, loader, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot = {"loss": 0.0, "dice": 0.0, "iou": 0.0}
    n = 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            b = x.size(0)
            tot["loss"] += loss.item() * b
            tot["dice"] += dice_score(logits, y) * b
            tot["iou"] += iou_score(logits, y) * b
            n += b
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="unet", choices=["unet", "resnet34", "resnet50"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    device = get_device()
    print("device:", device)
    split = patient_split(a.data)
    sets = {k: LGGDataset(list_pairs(a.data, v), train=(k == "train")) for k, v in split.items()}
    print({k: len(v) for k, v in sets.items()}, "slices")
    dl = {k: DataLoader(v, batch_size=a.bs, shuffle=(k == "train"), num_workers=a.workers) for k, v in sets.items()}

    model = build_model(a.arch, a.pretrained).to(device)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    os.makedirs("weights", exist_ok=True); os.makedirs("outputs", exist_ok=True)
    tag = a.arch + ("_pre" if a.pretrained else "")
    best, history = 0.0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, dl["train"], device, loss_fn, opt)
        va = run_epoch(model, dl["val"], device, loss_fn)
        sched.step(va["dice"])
        history.append({"epoch": ep, "train": tr, "val": va})
        print(f"ep {ep:02d} | train dice {tr['dice']:.3f} | val dice {va['dice']:.3f} iou {va['iou']:.3f} | {time.time()-t0:.0f}s")
        if va["dice"] > best:
            best = va["dice"]
            torch.save({"arch": a.arch, "state_dict": model.state_dict(), "val_dice": best}, f"weights/{tag}.pt")
    json.dump(history, open(f"outputs/history_{tag}.json", "w"), indent=2)
    print("best val dice:", round(best, 4), "->", f"weights/{tag}.pt")


if __name__ == "__main__":
    main()
