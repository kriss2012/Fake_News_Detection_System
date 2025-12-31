# Backend for Fake News Detector

Files:
- `train_model.py` - Train a TF-IDF + LogisticRegression pipeline from a CSV dataset and save `model.pkl`.
- `model.py` - Helper to load the saved model and make predictions.
- `app.py` - FastAPI app that accepts file uploads (PDF, image, TXT) and raw text and returns a prediction.

Quick start:

1. Create a Python environment and install dependencies:
```bash
python -m venv .venv

pip install -r backend/requirements.txt
```

2. Train model (assumes `backend/fake_review_dataset.csv` exists):
```bash
python backend/train_model.py backend/fake_review_dataset.csv --out backend/model.pkl
```

3. Run API server:
```bash
python backend/app.py
```

Notes:
- `pytesseract` requires Tesseract OCR installed on the system.
- `pdfplumber` provides PDF text extraction; for scanned PDFs OCR is used.
