from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# 🚀 LOAD MODELS ON STARTUP
# ==========================================
models = {
    "letters": load_model("models/FSL_Letters_model.keras"),
    "numbers": load_model("models/FSL_Numbers_Model.keras"),
    "ordinals": load_model("models/FSL_OrdinalNums_Model.keras"),
    "colors": load_model("models/FSL_Colors_model.keras"),
}

# ==========================================
# 🏷️ LABEL SETS
# ==========================================
label_sets = {
    "letters": [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X", "Y", "J", "Ñ", "NG", "Z"
    ],
    "numbers": [
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "10", "11", "12", "13", "14", "15", "16", "17",
        "18", "19", "20", "21"
    ],
    "ordinals": [
        "1st", "2nd", "3rd", "4th", "5th",
        "6th", "7th", "8th", "9th", "10th"
    ],
    "colors": [
        "BLACK", "BLUE", "BROWN", "GRAY", "GREEN",
        "ORANGE", "PINK", "RED", "VIOLET", "WHITE", "YELLOW"
    ],
}

# ==========================================
# ✋ HAND TYPE PER MODEL
# ==========================================
hand_type = {
    "letters": "one",
    "numbers": "one",
    "ordinals": "one",
    "colors": "two",  # ✅ two-hand model (FSL Colors trained with both hands)
}

# ==========================================
# ⚙️ FASTAPI SETUP
# ==========================================
app = FastAPI(title="SenyasFSL API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 replace with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📦 REQUEST SCHEMA
# ==========================================


class LandmarkRequest(BaseModel):
    left_hand: list | None = None
    right_hand: list | None = None
    sequence_length: int | None = None

# ==========================================
# 🧠 HELPER FUNCTIONS
# ==========================================


def preprocess_input(model_name: str, req: LandmarkRequest):
    """Prepares input array depending on one- or two-hand model type."""
    if hand_type[model_name] == "one":
        if req.right_hand is None:
            raise ValueError("Right hand data required for single-hand model.")
        seq = np.array(req.right_hand, dtype=np.float32)
    else:
        if req.left_hand is None or req.right_hand is None:
            raise ValueError(
                "Both left and right hand data required for two-hand model.")

        left = np.array(req.left_hand, dtype=np.float32)
        right = np.array(req.right_hand, dtype=np.float32)

        # ✅ Ensure both have same number of frames
        min_len = min(len(left), len(right))
        left, right = left[:min_len], right[:min_len]

        # ✅ Concatenate per frame: [left + right]
        seq = np.concatenate([left, right], axis=-1)

    # Add batch dimension for model input
    return np.expand_dims(seq, axis=0)


def predict_sequence(model_name: str, req: LandmarkRequest):
    """Runs prediction and returns only the top label (no confidence)."""
    model = models[model_name]
    labels = label_sets[model_name]
    seq = preprocess_input(model_name, req)

    prediction = model.predict(seq, verbose=0)[0]
    pred_idx = int(np.argmax(prediction))

    return {"prediction": labels[pred_idx]}

# ==========================================
# 🌐 ROUTES
# ==========================================


@app.get("/")
async def root():
    return {"message": "SenyasFSL API is running successfully!"}


@app.get("/models")
async def list_models():
    """Lists all loaded models for debugging."""
    return {"loaded_models": list(models.keys())}


@app.post("/predict/{model_name}")
async def predict(model_name: str, req: LandmarkRequest):
    """Runs prediction for the selected model."""
    if model_name not in models:
        return {"error": f"Unknown model '{model_name}'"}

    try:
        return predict_sequence(model_name, req)
    except Exception as e:
        return {"error": str(e)}
