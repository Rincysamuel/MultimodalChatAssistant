# backend/main.py

from fastapi import FastAPI, UploadFile, File
from backend.llava_model import LLaVAWrapper
from typing import List
from PIL import Image
import io

app = FastAPI(title="LLaVA Multimodal Backend")

# Initialize LLaVA model
llava = LLaVAWrapper(device='cpu')  # change to 'cuda' if GPU available

@app.get("/")
def root():
    return {"status": "LLaVA backend is running!"}

@app.post("/predict-text")
def predict_text(text: str):
    """
    Predict using only text input
    """
    output = llava.predict([text])
    return {"output": str(output)}

@app.post("/predict-image-text")
def predict_image_text(text: str, images: List[UploadFile] = File(...)):
    """
    Predict using text + image(s)
    """
    image_tensors = []
    for img_file in images:
        img = Image.open(io.BytesIO(img_file.file.read())).convert("RGB")
        image_tensors.append(img)  # You may need to transform to tensor depending on LLaVAWrapper

    output = llava.predict([text], image_inputs=image_tensors)
    return {"output": str(output)}
