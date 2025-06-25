import os
import psutil
import logging
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional
import threading
import csv

app = FastAPI(title="Offline Neural Machine Translation API", description="FastAPI backend for offline translation and LLM.")

MODEL_DIR = os.environ.get("TRANSLATOR_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
OPUS_MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
OPUS_BIG_MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-fr"
CROISSANT_MODEL_NAME = "croissantllm/CroissantLLMChat-v0.1"

os.makedirs(MODEL_DIR, exist_ok=True)

opus_model = None
opus_tokenizer = None
croissant_model = None
croissant_tokenizer = None

# Model download status
model_status = {
    "opus_mt": "not_loaded",
    "opus_mt_big": "not_loaded",
    "croissant_llm": "not_loaded"
}
model_lock = threading.Lock()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("translator-backend")

class TranslationRequest(BaseModel):
    text: str = Field(..., examples=["Hello world"])

class CroissantRequest(BaseModel):
    prompt: str = Field(..., examples=["What is the capital of France?"])

GLOSSARY_PATH = os.path.join(MODEL_DIR, "glossary.csv")
GLOSSARY_MAX_TERMS = 500

def load_glossary():
    glossary = []
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 2:
                    glossary.append({"source": row[0], "translation": row[1]})
    return glossary

def save_glossary(glossary):
    with open(GLOSSARY_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for entry in glossary:
            writer.writerow([entry["source"], entry["translation"]])

@app.on_event("startup")
def startup_event():
    threading.Thread(target=download_and_load_opus, daemon=True).start()
    threading.Thread(target=download_and_load_croissant, daemon=True).start()

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}

@app.get("/models/status", tags=["System"])
def models_status():
    return model_status.copy()

@app.get("/system/resources", tags=["System"])
def system_resources():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.5)
    return {
        "memory_total": mem.total,
        "memory_used": mem.used,
        "memory_percent": mem.percent,
        "cpu_percent": cpu
    }

@app.post("/translate/fr-en", tags=["Translation"])
def translate_fr_en(payload: TranslationRequest):
    text = payload.text
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'text' in request body.")
    try:
        with model_lock:
            if opus_model is None or opus_tokenizer is None:
                download_and_load_opus()
        input_ids = opus_tokenizer.encode(text, return_tensors="pt")
        output_ids = opus_model.generate(input_ids, max_length=256)
        translation = opus_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"translation": translation}
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Translation error: {e}")

@app.post("/croissant/generate", tags=["LLM"])
def croissant_generate(payload: CroissantRequest):
    prompt = payload.prompt
    if not prompt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'prompt' in request body.")
    try:
        with model_lock:
            if croissant_model is None or croissant_tokenizer is None:
                download_and_load_croissant()
        input_ids = croissant_tokenizer.encode(prompt, return_tensors="pt")
        output_ids = croissant_model.generate(input_ids, max_length=256)
        response = croissant_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        logger.error(f"Croissant LLM error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Croissant LLM error: {e}")

@app.post("/models/reload", tags=["System"])
def reload_models():
    """Reload all models without restarting the server."""
    try:
        with model_lock:
            download_and_load_opus(force=True)
            download_and_load_croissant(force=True)
        return {"status": "reloaded"}
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload error: {e}")

@app.get("/models/download/progress", tags=["System"])
def model_download_progress():
    return model_status.copy()

@app.post("/glossary/upload", tags=["Glossary"])
def upload_glossary(file: UploadFile = File(...)):
    glossary = load_glossary()
    reader = csv.reader((line.decode('utf-8') for line in file.file))
    count = len(glossary)
    for row in reader:
        if len(row) == 2 and count < GLOSSARY_MAX_TERMS:
            glossary.append({"source": row[0], "translation": row[1]})
            count += 1
        if count >= GLOSSARY_MAX_TERMS:
            break
    save_glossary(glossary)
    return {"count": len(glossary)}

@app.get("/glossary", tags=["Glossary"])
def get_glossary():
    return {"glossary": load_glossary()}

@app.post("/glossary/replace", tags=["Glossary"])
def replace_glossary(file: UploadFile = File(...)):
    glossary = []
    reader = csv.reader((line.decode('utf-8') for line in file.file))
    for row in reader:
        if len(row) == 2 and len(glossary) < GLOSSARY_MAX_TERMS:
            glossary.append({"source": row[0], "translation": row[1]})
        if len(glossary) >= GLOSSARY_MAX_TERMS:
            break
    save_glossary(glossary)
    return {"count": len(glossary)}

@app.post("/glossary/clear", tags=["Glossary"])
def clear_glossary():
    if os.path.exists(GLOSSARY_PATH):
        os.remove(GLOSSARY_PATH)
    return {"status": "cleared"}

@app.post("/croissant/review", tags=["LLM"])
def croissant_review(payload: CroissantRequest, use_glossary: Optional[bool] = Form(False)):
    prompt = payload.prompt
    glossary = load_glossary() if use_glossary else []
    glossary_text = "\n".join([f"{entry['source']} -> {entry['translation']}" for entry in glossary]) if glossary else ""
    full_prompt = prompt
    if glossary_text:
        full_prompt = f"Use the following glossary for translation preferences:\n{glossary_text}\n\n{prompt}"
    try:
        with model_lock:
            if croissant_model is None or croissant_tokenizer is None:
                download_and_load_croissant()
        input_ids = croissant_tokenizer.encode(full_prompt, return_tensors="pt")
        output_ids = croissant_model.generate(input_ids, max_length=256)
        response = croissant_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        logger.error(f"Croissant LLM review error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Croissant LLM review error: {e}")

def download_and_load_opus(force=False):
    global opus_model, opus_tokenizer
    try:
        if force or opus_model is None or opus_tokenizer is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model_status["opus_mt"] = "downloading"
            opus_tokenizer = AutoTokenizer.from_pretrained(OPUS_MODEL_NAME, cache_dir=MODEL_DIR)
            opus_model = AutoModelForSeq2SeqLM.from_pretrained(OPUS_MODEL_NAME, cache_dir=MODEL_DIR)
            model_status["opus_mt"] = "loaded"
            logger.info("Opus-MT model loaded.")
    except Exception as e:
        model_status["opus_mt"] = f"error: {e}"
        logger.error(f"Failed to load Opus-MT model: {e}")
        raise RuntimeError(f"Failed to load Opus-MT model: {e}")

def download_and_load_croissant(force=False):
    global croissant_model, croissant_tokenizer
    try:
        if force or croissant_model is None or croissant_tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_status["croissant_llm"] = "downloading"
            croissant_tokenizer = AutoTokenizer.from_pretrained(CROISSANT_MODEL_NAME, cache_dir=MODEL_DIR)
            croissant_model = AutoModelForCausalLM.from_pretrained(CROISSANT_MODEL_NAME, cache_dir=MODEL_DIR)
            model_status["croissant_llm"] = "loaded"
            logger.info("Croissant LLM loaded.")
    except Exception as e:
        model_status["croissant_llm"] = f"error: {e}"
        logger.error(f"Failed to load Croissant LLM: {e}")
        raise RuntimeError(f"Failed to load Croissant LLM: {e}")
