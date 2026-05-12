import os
import pickle
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spine Condition Prediction API",
    description=(
        "API untuk prediksi kondisi tulang belakang.\n\n"
        "Memprediksi 4 kondisi sekaligus:\n"
        "- **hyperkyphosis** (kyphosis berlebih di punggung atas)\n"
        "- **hyperlordosis** (lordosis berlebih di punggung bawah)\n"
        "- **re_hyperkyphosis** (kekambuhan hyperkyphosis)\n"
        "- **re_hyperlordosis** (kekambuhan hyperlordosis)\n\n"
        "Output: `0` = Negatif, `1` = Positif"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"[OK] Model berhasil dimuat dari: {MODEL_PATH}")
except FileNotFoundError:
    raise RuntimeError(f"Model tidak ditemukan di: {MODEL_PATH}")


# ── Schema ─────────────────────────────────────────────────────────────────
class SpineInput(BaseModel):
    sex: int = Field(..., ge=0, le=1, description="Jenis kelamin (0=Perempuan, 1=Laki-laki)")
    age: int = Field(..., gt=0, description="Umur (tahun)")
    height: float = Field(..., gt=0, description="Tinggi badan (cm)")
    weight: float = Field(..., gt=0, description="Berat badan (kg)")
    BMI: float = Field(..., gt=0, description="Body Mass Index")
    distC7S1: float = Field(..., description="Jarak C7-S1 (mm)")
    cerv: float = Field(..., description="Sudut servikal (derajat)")
    thorac: float = Field(..., description="Sudut torakal (derajat)")
    lumb: float = Field(..., description="Sudut lumbar (derajat)")
    KI: float = Field(..., description="Kyphosis Index")
    FC: float = Field(..., description="Flexion Contracture")
    FL: float = Field(..., description="Flexion Length")
    KI_pct: float = Field(..., alias="KI%", description="KI persentase (%)")
    FC_pct: float = Field(..., alias="FC%", description="FC persentase (%)")
    FL_pct: float = Field(..., alias="FL%", description="FL persentase (%)")

    model_config = {"populate_by_name": True}


class PredictionResult(BaseModel):
    hyperkyphosis: int = Field(..., description="0=Negatif, 1=Positif")
    hyperlordosis: int = Field(..., description="0=Negatif, 1=Positif")
    re_hyperkyphosis: int = Field(..., description="0=Negatif, 1=Positif")
    re_hyperlordosis: int = Field(..., description="0=Negatif, 1=Positif")

    # Interpretasi teks
    hyperkyphosis_label: str
    hyperlordosis_label: str
    re_hyperkyphosis_label: str
    re_hyperlordosis_label: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Status"])
def root():
    return {
        "status": "ok",
        "message": "Spine Condition Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Status"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(data: SpineInput):
    """
    Memprediksi kondisi tulang belakang berdasarkan parameter klinis.

    **Semua field wajib diisi.**
    """
    try:
        # Buat DataFrame dengan nama kolom yang sama persis seperti saat training
        input_df = pd.DataFrame([{
            "sex":      data.sex,
            "age":      data.age,
            "height":   data.height,
            "weight":   data.weight,
            "BMI":      data.BMI,
            "distC7S1": data.distC7S1,
            "cerv":     data.cerv,
            "thorac":   data.thorac,
            "lumb":     data.lumb,
            "KI":       data.KI,
            "FC":       data.FC,
            "FL":       data.FL,
            "KI%":      data.KI_pct,
            "FC%":      data.FC_pct,
            "FL%":      data.FL_pct,
        }])

        prediction = model.predict(input_df)

        # prediction shape: (1, 4) — [hyperkyphosis, hyperlordosis, re_hyperkyphosis, re_hyperlordosis]
        hk  = int(prediction[0][0])
        hl  = int(prediction[0][1])
        rhk = int(prediction[0][2])
        rhl = int(prediction[0][3])

        label = lambda v: "Positif" if v == 1 else "Negatif"

        return PredictionResult(
            hyperkyphosis=hk,
            hyperlordosis=hl,
            re_hyperkyphosis=rhk,
            re_hyperlordosis=rhl,
            hyperkyphosis_label=label(hk),
            hyperlordosis_label=label(hl),
            re_hyperkyphosis_label=label(rhk),
            re_hyperlordosis_label=label(rhl),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
