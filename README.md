# Spine Condition Prediction API

REST API untuk memprediksi kondisi tulang belakang menggunakan Machine Learning.

## 📋 Deskripsi

API ini memprediksi 4 kondisi tulang belakang secara sekaligus:
| Label | Deskripsi |
|---|---|
| `hyperkyphosis` | Kyphosis berlebih di punggung atas |
| `hyperlordosis` | Lordosis berlebih di punggung bawah |
| `re_hyperkyphosis` | Kekambuhan hyperkyphosis |
| `re_hyperlordosis` | Kekambuhan hyperlordosis |

**Output:** `0` = Negatif, `1` = Positif

## 🚀 Cara Menjalankan Secara Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan server
python app.py
```

Server berjalan di: `http://localhost:8000`  
Dokumentasi API: `http://localhost:8000/docs`

## 📡 Endpoint

### `GET /`
Health check.

### `GET /health`
Status server.

### `POST /predict`
Prediksi kondisi tulang belakang.

**Request Body:**
```json
{
  "sex": 1,
  "age": 35,
  "height": 175.0,
  "weight": 70.0,
  "BMI": 22.86,
  "distC7S1": 500.0,
  "cerv": 50.0,
  "thorac": -15.0,
  "lumb": 25.0,
  "KI": 52.0,
  "FC": 65.0,
  "FL": 40.0,
  "KI%": 10.4,
  "FC%": 13.0,
  "FL%": 8.0
}
```

**Response:**
```json
{
  "hyperkyphosis": 0,
  "hyperlordosis": 0,
  "re_hyperkyphosis": 0,
  "re_hyperlordosis": 0,
  "hyperkyphosis_label": "Negatif",
  "hyperlordosis_label": "Negatif",
  "re_hyperkyphosis_label": "Negatif",
  "re_hyperlordosis_label": "Negatif"
}
```

## 📦 Struktur Folder

```
spine-api/
├── app.py              # FastAPI server
├── requirements.txt    # Python dependencies
├── Procfile            # Untuk Railway/Heroku
├── .gitignore
├── README.md
└── models/
    ├── model.pkl               # Model utama
    ├── scaler_*.pkl            # StandardScaler
    └── pipeline_metadata_*.json
```

## 🔧 Keterangan Field Input

| Field | Tipe | Keterangan |
|---|---|---|
| `sex` | int | 0=Perempuan, 1=Laki-laki |
| `age` | int | Umur dalam tahun |
| `height` | float | Tinggi badan (cm) |
| `weight` | float | Berat badan (kg) |
| `BMI` | float | Body Mass Index |
| `distC7S1` | float | Jarak C7 ke S1 (mm) |
| `cerv` | float | Sudut servikal (derajat) |
| `thorac` | float | Sudut torakal (derajat) |
| `lumb` | float | Sudut lumbar (derajat) |
| `KI` | float | Kyphosis Index |
| `FC` | float | Flexion Contracture |
| `FL` | float | Flexion Length |
| `KI%` | float | KI persentase |
| `FC%` | float | FC persentase |
| `FL%` | float | FL persentase |

## 🛠️ Model Info

| Label | Model | Accuracy | F1 Score |
|---|---|---|---|
| hyperkyphosis | CatBoost | 83.5% | 75.9% |
| hyperlordosis | KNN | 86.5% | 79.7% |
| re_hyperkyphosis | AdaBoost | 92.2% | 89.0% |
| re_hyperlordosis | KNN (Specialist) | 93.5% | 91.0% |
