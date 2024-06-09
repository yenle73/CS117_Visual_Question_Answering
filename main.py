from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
from io import BytesIO
import uvicorn

app = FastAPI()

# Load the model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Serve the static HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(image: UploadFile = File(...), question: str = Form(...)):
    # Load image from the uploaded file
    image_bytes = await image.read()
    pil_image = Image.open(BytesIO(image_bytes))

    # Prepare inputs
    encoding = processor(pil_image, question, return_tensors="pt")

    # Forward pass
    outputs = model(**encoding)
    logits = outputs.logits

    # Get predicted answer and probabilities
    probabilities = torch.softmax(logits, dim=-1)
    idx = logits.argmax(-1).item()
    predicted_answer = model.config.id2label[idx]
    predicted_prob = probabilities[0, idx].item()

    # Prepare the response
    response = {
        "predicted_answer": predicted_answer,
        "predicted_probability": predicted_prob,
        "all_answers": [
            {"answer": model.config.id2label[i], "probability": probabilities[0, i].item()}
            for i in range(len(model.config.id2label))
        ]
    }
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
