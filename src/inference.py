import cv2, numpy as np, torch
from .dataset import IMG_SIZE, MEAN, STD
from .model import build_model


def load_model(path="weights/best.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict_mask(model, rgb, device="cpu", threshold=0.5):
    """rgb: HxWx3 uint8. Returns (mask HxW uint8 at original size, tumor_area_fraction)."""
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = cv2.resize((prob > threshold).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask, float(mask.mean())


def overlay(rgb, mask, color=(255, 0, 0), alpha=0.45):
    out = rgb.copy()
    out[mask > 0] = ((1 - alpha) * out[mask > 0] + alpha * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 1)
    return out
