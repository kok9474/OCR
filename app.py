from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from transformers import AutoModelForVision2Seq, TrOCRProcessor
import io
import logging
import os
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
processor = None
device = None

def _find_local_model_path() -> str | None:
    for path in ("./model_plus", "./model", "./model_output"):
        if os.path.exists(path):
            return path
    return None

# 서버 초기화
@app.on_event("startup")
async def load_model():
    global model, processor, device
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        local_path = _find_local_model_path()
        if local_path:
            logger.info(f"Loading fine-tuned model from {local_path}")
            model = AutoModelForVision2Seq.from_pretrained(local_path)
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            model_source = "local"
        else:
            logger.info("No local model found. Loading base TrOCR model...")
            model = AutoModelForVision2Seq.from_pretrained("microsoft/trocr-base-printed")
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            model_source = "base"

        model.to(device).eval()
        logger.info(f"Model loaded successfully from: {model_source}")

    except Exception as e:
        logger.exception("Failed to load model")
        raise e


def preprocess_image(image: Image.Image) -> Image.Image:
    """Lightweight, safe defaults for OCR."""
    image = image.convert("RGB")
    image = ImageOps.pad(image, (384, 384), method=Image.BICUBIC, color=(255, 255, 255))
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def _generate_text_from_image(img: Image.Image) -> str:
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    processed = preprocess_image(img)
    pixel_values = processor(processed, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=10,
            num_beams=4,
            decoder_start_token_id=processor.tokenizer.cls_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "OCR Service",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "model_source": "local" if _find_local_model_path() else "base",
    }


# 단일 이미지
@app.post("/ocr/extract")
async def extract_text_from_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        text = _generate_text_from_image(img)
        return {"success": True, "ocr_text": text, "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OCR extraction failed")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


# 여러 이미지 일괄
@app.post("/ocr/batch")
async def extract_text_batch(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        try:
            img = Image.open(io.BytesIO(await f.read()))
            text = _generate_text_from_image(img)
            results.append({"filename": f.filename, "success": True, "ocr_text": text})
        except Exception as e:
            results.append({"filename": f.filename, "success": False, "error": str(e)})
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000, reload=False)
