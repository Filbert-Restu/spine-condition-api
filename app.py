import os
import pickle
import numpy as np
import pandas as pd
import uvicorn
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score, accuracy_score

# ==============================================================================
# CUSTOM CLASSES (Required for Pickle Deserialization)
# ==============================================================================

class SpinalSpecialistModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, feature_map):
        self.base_estimator = base_estimator
        self.feature_map = feature_map
        self.models_ = {} 
        self.target_names_ = list(feature_map.keys())

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X harus berupa Pandas DataFrame.")
        
        for target in self.target_names_:
            required_features = self.feature_map[target]
            X_subset_df = X[required_features]
            y_subset_raw = y[target]
            X_final = X_subset_df.values
            y_final = np.array(y_subset_raw).ravel()
            model = clone(self.base_estimator)
            model.fit(X_final, y_final)
            self.models_[target] = model
        return self

    def predict(self, X):
        predictions = {}
        for target in self.target_names_:
            required_features = self.feature_map[target]
            X_subset_df = X[required_features]
            X_final = X_subset_df.values
            predictions[target] = self.models_[target].predict(X_final)
        return pd.DataFrame(predictions).values

    def predict_proba(self, X):
        probabilities = []
        for target in self.target_names_:
            required_features = self.feature_map[target]
            X_subset_df = X[required_features]
            X_final = X_subset_df.values
            if hasattr(self.models_[target], "predict_proba"):
                probabilities.append(self.models_[target].predict_proba(X_final))
            else:
                preds = self.models_[target].predict(X_final)
                prob = np.zeros((len(preds), 2))
                for i, p in enumerate(preds):
                    prob[i, int(p)] = 1.0
                probabilities.append(prob)
        return probabilities

class OptimizedSpinalPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline yang menggunakan model terbaik untuk setiap label.
    Setiap label diprediksi oleh model yang memiliki performa terbaik untuk label tersebut.
    """

    def __init__(self, label_models, feature_map, scaler=None):
        self.label_models = label_models
        self.feature_map = feature_map
        self.scaler = scaler
        self.target_names_ = list(label_models.keys())

    def fit(self, X, y):
        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X harus berupa Pandas DataFrame")

        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X.copy()

        predictions = {}
        for label in self.target_names_:
            model_info = self.label_models[label]
            model = model_info['model']

            if 'SpinalSpecialistModel' in model.__class__.__name__:
                pred_full = model.predict(X_scaled)
                label_idx = self.target_names_.index(label)
                predictions[label] = pred_full[:, label_idx]
            else:
                required_features = self.feature_map[label]
                X_subset = X_scaled[required_features]
                predictions[label] = model.predict(X_subset.values)

        result = np.column_stack([predictions[label] for label in self.target_names_])
        return result

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X harus berupa Pandas DataFrame")

        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X.copy()

        probabilities = {}
        for label in self.target_names_:
            model_info = self.label_models[label]
            model = model_info['model']

            try:
                if 'SpinalSpecialistModel' in model.__class__.__name__:
                    proba_full = model.predict_proba(X_scaled)
                    label_idx = self.target_names_.index(label)
                    probabilities[label] = proba_full[label_idx][:, 1]
                else:
                    required_features = self.feature_map[label]
                    X_subset = X_scaled[required_features]
                    proba = model.predict_proba(X_subset.values)
                    probabilities[label] = proba[:, 1]
            except AttributeError:
                probabilities[label] = np.zeros(X.shape[0])

        return probabilities

    def get_model_info(self):
        info = {}
        for label in self.target_names_:
            model_info = self.label_models[label]
            info[label] = {
                'model_name': model_info.get('model_name', str(model_info['model'].__class__.__name__)),
                'f1_score': model_info.get('f1_score', 0),
                'accuracy': model_info.get('accuracy', 0),
                'features_used': self.feature_map.get(label, [])
            }
        return info

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load Model ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info(f"[OK] Model berhasil dimuat dari: {MODEL_PATH}")
    else:
        logger.error(f"[ERROR] File model tidak ditemukan: {MODEL_PATH}")
except Exception as e:
    logger.error(f"[ERROR] Gagal memuat model: {str(e)}")
    logger.error(traceback.format_exc())
    model = None


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
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Check server logs for errors."
        )

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
