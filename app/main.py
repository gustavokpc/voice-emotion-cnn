from __future__ import annotations

import json
from functools import lru_cache
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.audio import to_mel
from app.model import EmotionCNN, TransferEmotionCNN

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS, UPLOADS = ROOT / "artifacts", ROOT / "uploads"
MODEL_ARCHITECTURES = ("scratch", "transfer")
UPLOADS.mkdir(exist_ok=True)
app = FastAPI(title="Voice Emotion CNN")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

@lru_cache(maxsize=1)
def load_model(architecture: str) -> tuple[torch.nn.Module, list[str]]:
    if architecture not in MODEL_ARCHITECTURES:
        raise HTTPException(400, "Unknown model architecture. Choose scratch or transfer.")

    labels_path = ARTIFACTS / "labels.json"
    weights_path = ARTIFACTS / f"model-{architecture}.pt"
    if not labels_path.exists() or not weights_path.exists():
        raise HTTPException(503, f"The {architecture} model is unavailable. Run its training first.")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    model = TransferEmotionCNN(len(labels), pretrained=False) if architecture == "transfer" else EmotionCNN(len(labels))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, labels


def available_models() -> list[str]:
    return [architecture for architecture in MODEL_ARCHITECTURES if (ARTIFACTS / f"model-{architecture}.pt").exists()]


def model_statistics(architecture: str) -> dict:
    statistics_path = ARTIFACTS / f"evaluation-{architecture}.json"
    return json.loads(statistics_path.read_text(encoding="utf-8")) if statistics_path.exists() else {}

@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")

@app.get("/health")
def health() -> dict[str, bool]:
    return {"trained_model_available": bool(available_models())}


@app.get("/models")
def models() -> dict[str, list[dict]]:
    return {
        "models": [
            {"architecture": architecture, "statistics": model_statistics(architecture)}
            for architecture in available_models()
        ]
    }

@app.post("/predict")
async def predict(audio: UploadFile = File(...), architecture: str = Form("transfer")) -> dict:
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    source = UPLOADS / f"{uuid.uuid4()}{suffix}"
    with source.open("wb") as output: shutil.copyfileobj(audio.file, output)
    try:
        model, labels = load_model(architecture)
        features = torch.from_numpy(to_mel(source)).unsqueeze(0)
        with torch.inference_mode(): probabilities = torch.softmax(model(features), dim=1)[0].numpy()
        ranked = np.argsort(probabilities)[::-1]
        return {"architecture": architecture, "emotion": labels[int(ranked[0])], "confidence": round(float(probabilities[ranked[0]]), 4), "scores": {labels[int(i)]: round(float(probabilities[i]), 4) for i in ranked}, "spectrogram": features[0, 0].tolist()}
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        source.unlink(missing_ok=True)
