from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llmops_api")

app = FastAPI(
    title="LLMOps Inference API",
    description="A production-ready API for serving a large language model.",
    version="1.0.0"
)

# Define request and response models
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 150
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str

# Initialize the text-generation pipeline (using GPT-2 for demo; replace with your LLM)
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = pipeline("text-generation", model="gpt2", device=0 if device=="cuda" else -1)

# Middleware for basic request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}

@app.post("/generate", response_model=GenerationResponse, tags=["Inference"])
def generate_text(req: GenerationRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty")
    try:
        outputs = generator(
            req.prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            num_return_sequences=1
        )
        generated_text = outputs[0]["generated_text"]
        return GenerationResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Model generation failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
