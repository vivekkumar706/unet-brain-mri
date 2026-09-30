"""FastAPI REST API (/predict, /health) with the Gradio demo mounted at '/'."""
import io, os
import cv2, gradio as gr, numpy as np, torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from src.inference import load_model, overlay, predict_mask

WEIGHTS = os.getenv("WEIGHTS", "weights/best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(WEIGHTS, DEVICE)

api = FastAPI(title="Brain MRI Tumor Segmentation", version="1.0")


@api.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@api.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        rgb = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))
    except Exception:
        raise HTTPException(400, "Could not read image")
    mask, frac = predict_mask(model, rgb, DEVICE)
    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay(rgb, mask), cv2.COLOR_RGB2BGR))
    return Response(png.tobytes(), media_type="image/png", headers={"X-Tumor-Area-Fraction": f"{frac:.5f}"})


def demo_fn(img, threshold):
    if img is None:
        return None, "Upload an MRI slice."
    rgb = np.array(img.convert("RGB"))
    mask, frac = predict_mask(model, rgb, DEVICE, threshold)
    msg = "No tumor region detected." if mask.sum() == 0 else f"Predicted tumor covers {frac*100:.2f}% of the slice."
    return overlay(rgb, mask), msg + "\n\nResearch demo only, not for clinical use."


ui = gr.Interface(
    fn=demo_fn,
    inputs=[gr.Image(type="pil", label="Brain MRI slice"), gr.Slider(0.1, 0.9, 0.5, step=0.05, label="Threshold")],
    outputs=[gr.Image(label="Predicted tumor overlay"), gr.Textbox(label="Result")],
    title="Brain Tumor MRI Segmentation (U-Net)",
    description="Upload an MRI slice (e.g. from the LGG dataset) to see the predicted tumor mask.",
    flagging_mode="never",
)
app = gr.mount_gradio_app(api, ui, path="/")
