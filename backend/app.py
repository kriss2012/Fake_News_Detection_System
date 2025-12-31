import io
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

import sys
# Ensure the backend package directory is on sys.path so local imports work
sys.path.insert(0, os.path.dirname(__file__))
import model as model_module

import pdfplumber
from PIL import Image
import pytesseract

app = FastAPI(title='Fake News Predictor')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (if present)
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
if os.path.isdir(FRONTEND_DIR):
    app.mount('/static', StaticFiles(directory=FRONTEND_DIR), name='static')


@app.get('/', include_in_schema=False)
def root():
    index = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index):
        return FileResponse(index, media_type='text/html')
    # fallback to docs if no index.html
    return RedirectResponse(url='/docs')

# Load model at startup
MODEL = None
@app.on_event('startup')
def load_model():
    global MODEL
    try:
        MODEL = model_module.load_model()
        print('Model loaded.')
    except Exception as e:
        print('Warning: could not load model at startup:', e)


def extract_text_from_pdf_bytes(data: bytes) -> str:
    text = ''
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
    except Exception:
        text = ''
    return text


def extract_text_from_image_bytes(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data)).convert('RGB')
        text = pytesseract.image_to_string(img)
        return text
    except Exception:
        return ''


@app.post('/predict-file')
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or ''
    lower = filename.lower()

    text = ''
    if lower.endswith('.pdf'):
        text = extract_text_from_pdf_bytes(content)
        if not text:
            # fallback: render first page with OCR
            try:
                text = extract_text_from_image_bytes(content)
            except Exception:
                text = ''
    elif lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        text = extract_text_from_image_bytes(content)
    elif lower.endswith('.txt'):
        try:
            text = content.decode('utf-8', errors='ignore')
        except Exception:
            text = ''
    else:
        # Attempt to treat as text
        try:
            text = content.decode('utf-8', errors='ignore')
        except Exception:
            text = ''

    if not text.strip():
        return JSONResponse({'error': 'No text extracted from file'}, status_code=400)

    results = model_module.predict_texts([text], model=MODEL)
    return results[0]


@app.post('/predict-text')
async def predict_text(text: str = Form(...)):
    if not text or not text.strip():
        return JSONResponse({'error': 'Empty text'}, status_code=400)
    results = model_module.predict_texts([text], model=MODEL)
    return results[0]


if __name__ == '__main__':
    # allow overriding bind host/port with env vars; default to localhost for safety
    host = os.environ.get('HOST', '127.0.0.1')
    port = int(os.environ.get('PORT', '8000'))
    print(f"Starting server on http://{host}:{port} (use 127.0.0.1 for local access)")
    uvicorn.run(app, host=host, port=port)
#.venv\Scripts\Activate.ps1 to activate the venv