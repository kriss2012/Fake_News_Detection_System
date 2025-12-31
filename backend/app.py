import io
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, AsyncIterator
import uvicorn
import sys

# Ensure the backend package directory is on sys.path so local imports work
sys.path.insert(0, os.path.dirname(__file__))
import model as model_module

import pdfplumber
from PIL import Image
import pytesseract

# Lifespan manager for startup/shutdown events (replaces deprecated @app.on_event)
MODEL = None

async def startup_event() -> None:
    """Load model on startup"""
    global MODEL
    try:
        MODEL = model_module.load_model()
        print('Model loaded successfully.')
    except Exception as e:
        print(f'Warning: could not load model at startup: {e}')

async def shutdown_event() -> None:
    """Cleanup on shutdown"""
    global MODEL
    print('Shutting down server...')

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan event handler - startup and shutdown"""
    await startup_event()
    yield
    await shutdown_event()

# Create FastAPI app with lifespan
app = FastAPI(title='Fake News Predictor', lifespan=lifespan)

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

# ✅ FIXED: Root endpoint that supports HEAD for Render health checks
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)  # Explicit HEAD support for Render
async def root():
    """Root endpoint - serves frontend or API docs"""
    index = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index):
        return FileResponse(index, media_type='text/html')
    # fallback to docs if no index.html
    return RedirectResponse(url='/docs')

# ✅ NEW: Dedicated health check endpoint for Render
@app.get("/health", include_in_schema=False)
@app.head("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint for Render deployment"""
    return {"status": "healthy", "service": "Fake News Predictor API"}

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
    """Predict fake news from uploaded file (PDF, image, text)"""
    if MODEL is None:
        return JSONResponse({'error': 'Model not loaded'}, status_code=503)
    
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
    """Predict fake news from text input"""
    if MODEL is None:
        return JSONResponse({'error': 'Model not loaded'}, status_code=503)
    
    if not text or not text.strip():
        return JSONResponse({'error': 'Empty text'}, status_code=400)
    
    results = model_module.predict_texts([text], model=MODEL)
    return results[0]

if __name__ == '__main__':
    # ✅ FIXED: Proper Render port/host binding
    host = os.environ.get('HOST', '0.0.0.0')  # Use 0.0.0.0 for Render/Docker
    port = int(os.environ.get('PORT', '8000'))
    print(f"Starting server on http://{host}:{port} (use 127.0.0.1 for local access)")
    uvicorn.run(
        "app:app",  # Note: use module:app format for proper reload
        host=host, 
        port=port,
        reload=False  # Disable reload in production
    )
