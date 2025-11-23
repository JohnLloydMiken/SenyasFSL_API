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
    "family_relationship": load_model("models/FSL_Family_Relationship_model.keras"),
    "socialization": load_model("models/FSL_Socialization_model.keras"),
    "timeExpression_daysOfWeeks": load_model("models/FSL_TimeExpression_DaysOfWeek_model.keras"),
    "timeExpression_months": load_model("models/FSL_TimeExpression_Months_model.keras"),
    "time_weather": load_model("models/FSL_Time_Weather_model.keras"),
    "colors_house": load_model("models/FSL_Colors_House_model.keras"),
    "occupation_subjects": load_model("models/FSL_Occupation_Subjects_model.keras"),
   

}

# ==========================================
# 🏷️ LABEL SETS
# ==========================================
label_sets = {
    "letters": [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X", "Y", "J", "Ñ", "Ng", "Z"
    ],
    "numbers": [
        "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
        "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
        "Seventeen", "Eighteen", "Nineteen", "Twenty", "Twenty-one"
    ],
    "ordinals": [
        "First", "Second", "Third", "Fourth", "Fifth",
        "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"
    ],
    "family_relationship": [
        "Aunt", "Beautiful", "Boyfriend", "Brother", "Child",
        "Classmate", "Cousin", "Crush", "Family", "Father", "Friend",
        "Girlfriend", "Godfather", "Godmother", "Grandchild", "Grandfather",
        "Grandmother", "Handsome", "Husband", "Like", "Love", "Mother", "Nephew",
        "Niece", "Sister", "Uncle", "Wife"
    ],
    "socialization": [
        "Again", "Bye", "Deaf", "Don't know", "Don't understand", "Excuseme", "Filipino",
        "Hard of hearing", "Hearing", "Hello", "How", "Know", "Language", "No", "Okay",
        "Please", "Ready", "Sign", "Sorry", "Stop", "Understand", "Wait", "What",
        "When", "Where", "Who", "Why", "Yes"
    ],
    "timeExpression_daysOfWeeks": [
        "Early", "Friday", "Hour", "Last", "Late",
        "Minutes", "Monday", "Never", "Once", "Recent",
        "Saturday", "Seconds", "See you", "Sometime", "Soon",
        "Sunday", "Thursday", "Today", "Tuesday", "Twice",
        "Wednesday", "Week"
    ],
    "timeExpression_months": [
        "April", "August", "Before", "Calendar", "December",
        "Evening", "February", "January", "July", "June",
        "Later", "March", "May", "Months", "Morning",
        "Next week", "Night", "Noon", "November", "October",
        "Past", "September", "Tomorrow", "Year", "Yesterday"
    ],
    "time_weather": [
        "10:00 AM", "10:00 PM", "11:00 AM", "11:00 PM", "12:00 AM",
        "12:00 PM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM",
        "5:00 PM", "6:00 AM", "6:00 PM", "7:00 AM", "7:00 PM",
        "8:00 AM", "8:00 PM", "9:00 AM", "9:00 PM",
        "Bright", "Clear sky", "Cloud", "Cold",
        "Fog", "Hot", "Rainbow", "Rainy",
        "Smog", "Sun", "Warm", "Wet"
    ],
    "colors_house": [
        "Apartment", "Balcony", "Bedroom", "Black", "Blue",
        "Brown", "Closet", "Comfort Room", "Dining area", "First floor",
        "Garage", "Gray", "Green", "Hallway", "House",
        "Kitchen area", "Living room", "Nipa hut", "Orange", "Pink",
        "Red", "Room", "Second floor", "Storage room", "Third floor",
        "Violet", "White", "Yellow"
    ],
    "occupation_subjects": [
        "Araling panlipunan", "Architect", "Assignment", "Bakery", "Cementery",
        "Chef", "Church", "Clinic", "Computer", "Doctor",
        "English", "Exam", "Farmers market", "Filipino", "Firestation",
        "Grocery", "Hekasi", "Hospital", "Hotel", "Jail",
        "Makabayan", "Math", "Occupation", "Philosophy", "Photographer",
        "Quiz", "Science", "Student", "Teacher", "Test"
    ]

}

# ==========================================
# ✋ HAND TYPE PER MODELs
# ==========================================
hand_type = {
    "letters": "one",
    "numbers": "one",
    "ordinals": "one",
    # ✅ two-hand model (FSL Colors trained with both hands)
    "family_relationship": "two",
    "socialization": "two",
    "timeExpression_daysOfWeeks": "two",
    "timeExpression_months": "two",
    "time_weather": "two",
    "colors_house": "two",
    "occupation_subjects": "two",

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
    """Runs prediction and returns top label with confidence (2 decimal places)."""
    model = models[model_name]
    labels = label_sets[model_name]
    seq = preprocess_input(model_name, req)

    # Get prediction probabilities
    prediction = model.predict(seq, verbose=0)[0]

    # Get index and confidence
    pred_idx = int(np.argmax(prediction))
    confidence = float(prediction[pred_idx])

    # Return both label and confidence (2 decimal points)
    return {
        "prediction": labels[pred_idx],
        "confidence": round(confidence * 100, 2)  # percentage format
    }

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
