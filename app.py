from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import numpy as np
import asyncio
import os
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import logging

# Setup logger
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

# Optional: ensure logs print during development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# File paths
best_model_path = "./models/best_model.keras"
label_encoder_path = "./models/label_encoder.npy"
scaler_params_path = "./models/scaler_params.npy"

load_dotenv()
client = Groq()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################
# Alphabet Classifier Setup
########################
class AlphabetClassifier:
    def __init__(self, model_path=best_model_path):
        logger.info("Loading model and preprocessing data...")
        self.model = load_model(model_path)
        self.scaler_params = np.load(scaler_params_path, allow_pickle=True).item()
        self.labels = np.load(label_encoder_path, allow_pickle=True)

        self.scaler_mean = self.scaler_params["mean"]
        self.scaler_scale = self.scaler_params["scale"]
        logger.info("Model and label encoders loaded successfully.")

    def predict(self, sensor_data):
        X = np.array([sensor_data], dtype=np.float32)
        X_scaled = (X - self.scaler_mean) / self.scaler_scale
        proba = self.model.predict(X_scaled, verbose=0)[0]
        pred_idx = np.argmax(proba)
        return self.labels[pred_idx], float(proba[pred_idx])

# Initialize classifier once
classifier = AlphabetClassifier()

# Pydantic model
class SensorDataRequest(BaseModel):
    sensor_data: list

class ChatRequest(BaseModel):
    user_message: str

async def stream_response(user_message: str):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a chatbot specialized exclusively in topics related to sign language."},
            {"role": "user", "content": user_message},
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content
        await asyncio.sleep(0.01)

@app.get("/")
def welcome():
    return {"message": "Welcome to the Sign Language API! (Classifier + Chatbot)"}

@app.post("/predict")
def predict(sensor_request: SensorDataRequest):
    try:
        logger.info(f"Received sensor data: {sensor_request.sensor_data}")
        letter, confidence = classifier.predict(sensor_request.sensor_data)
        logger.info(f"Predicted letter: {letter}, Confidence: {confidence:.2%}")
        return {"predicted_letter": letter, "confidence": f"{confidence:.2%}"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_response(request.user_message),
        media_type="text/event-stream"
    )

# Dev entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
