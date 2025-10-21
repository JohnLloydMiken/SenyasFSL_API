from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

# ====== LOAD MODELS ON STARTUP ======
models = {
    "letters": load_model("models/FSL_Letters_model.keras"),
    "numbers": load_model("models/FSL_Numbers_Model.keras"),
    "ordinal": load_model("models/FSL_OrdinalNums_Model.keras"),
    "colors": load_model("models/FSL_Colors_model.keras"),  # 👈 NEW
}

# ====== LABEL SETS ======
label_sets = {
    "letters": [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X", "Y", "J", "Ñ", "NG", "Z"
    ],
    "numbers": ["1", "2", "3", "4", "5", "6", "7", "8", "9",
                "10", "11", "12", "13", "14", "15", "16", "17",
                "18", "19", "20", "21"],
    "ordinals": ["1st","2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",],
    "colors": [  # 👈 Add your exact trained labels here
        "Red", "Blue", "Green", "Yellow", "Black",
        "White", "Brown", "Pink", "Orange", "Purple", "Gray"
    ],
}

# ====== HAND TYPE ======
hand_type = {
    "letters": "one",
    "numbers": "one",
    "ordinals": "one",
    "colors": "two", 
}

# ====== FASTAPI SETUP ======
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Change later to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== DATA MODEL ======
class LandmarkRequest(BaseModel):
    left_hand: list | None = None
    right_hand: list | None = None
    sequence_length: int | None = None

# ====== HELPER FUNCTIONS ======
def preprocess_input(model_name, req: LandmarkRequest):
    """Prepare the sequence data for model prediction."""
    if hand_type[model_name] == "one":
        if req.right_hand is None:
            raise ValueError("Right hand data required for single-hand model.")
        seq = np.array(req.right_hand, dtype=np.float32)
    else:
        if req.left_hand is None or req.right_hand is None:
            raise ValueError(
                "Both left and right hand data required for two-hand model."
            )
        seq = np.concatenate([req.left_hand, req.right_hand], axis=-1)

    seq = np.expand_dims(seq, axis=0)
    return seq


def predict_sequence(model_name, req: LandmarkRequest):
    """Run prediction for a given model."""
    model = models[model_name]
    labels = label_sets[model_name]
    seq = preprocess_input(model_name, req)

    prediction = model.predict(seq, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        "prediction": labels[predicted_index],
        "confidence": round(confidence, 4)
    }

# ====== ROUTES ======
@app.get("/")
async def root():
    return {"message": "SenyasFSL API is running!"}


@app.post("/predict/{model_name}")
async def predict(model_name: str, req: LandmarkRequest):
    if model_name not in models:
        return {"error": f"Unknown model '{model_name}'"}
    try:
        return predict_sequence(model_name, req)
    except Exception as e:
        return {"error": str(e)}
