---
title: Brain MRI Tumor Segmentation
emoji: 🧠
sdk: docker
app_port: 7860
---

# Brain Tumor MRI Segmentation (U-Net)

Pixel-level tumor segmentation on brain MRI using a U-Net built from scratch in PyTorch, compared against a
pretrained-encoder U-Net. Patient-level splits, Dice+BCE loss, augmentation, error analysis, and a deployed demo
(FastAPI + Gradio + Docker).

**Live demo:** _add your Hugging Face Space link_ | **Dataset:** [LGG MRI Segmentation (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

## Results (held-out test patients)
_Fill from `outputs/results.json` after running evaluation._

| Model | Dice | IoU | FP rate on healthy slices |
|---|---|---|---|
| U-Net (scratch) | | | |
| U-Net + ResNet34 (ImageNet) | | | |

![samples](outputs/samples/sample_0.png)
_Left to right: MRI, ground truth (green), prediction (red)._

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download the dataset from Kaggle; unzip so data/kaggle_3m/<patient>/*.tif exists
# 2. Train both models (MPS / CUDA / CPU chosen automatically)
python -m src.train --data data/kaggle_3m --arch unet --epochs 40
python -m src.train --data data/kaggle_3m --arch resnet34 --pretrained --epochs 25
# 3. Evaluate on held-out patients and save overlay samples
python -m src.evaluate --data data/kaggle_3m
# 4. Choose the better checkpoint for deployment
cp weights/resnet34_pre.pt weights/best.pt      # or weights/unet.pt
```

## Run the app
```bash
uvicorn app.main:app --port 7860      # demo: http://localhost:7860 , API docs: /docs
curl -F "file=@scan.tif" localhost:7860/predict -o overlay.png
```

### Docker
```bash
docker build -t brain-mri-seg . && docker run -p 7860:7860 brain-mri-seg
```

## Deploy free on Hugging Face Spaces
1. Create a new Space with SDK **Docker** (blank template).
2. Clone the Space repo and copy this project into it, including `weights/best.pt` (`git lfs track "*.pt"`).
3. `git add . && git commit -m "deploy" && git push`. The Space builds and serves on port 7860.

## Design notes
- **Patient-level split** prevents slice leakage (neighbouring slices of one patient are near-duplicates).
- **Dice + BCE** loss handles the heavy foreground/background imbalance.
- **Error analysis:** small tumors, blurry boundaries, false positives on tumor-free slices.
- **Disclaimer:** research demo, not a medical device.
