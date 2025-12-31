import os
import joblib
from typing import List

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def load_model(path: str = None):
    p = path or MODEL_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f'Model not found at {p}. Train with train_model.py')
    model = joblib.load(p)
    return model


def predict_texts(texts: List[str], model=None):
    if model is None:
        model = load_model()
    probs = model.predict_proba(texts)[:, 1]
    preds = model.predict(texts)
    results = []
    for p, prob in zip(preds, probs):
        # p is 0 (real) or 1 (fake)
        label_str = 'fake' if int(p) == 1 else 'real'
        # confidence: probability in favor of the chosen class (0..1)
        confidence_score = float(prob) if int(p) == 1 else float(1.0 - prob)
        confidence_pct = round(confidence_score * 100.0, 2)

        # verdict thresholds (can be adjusted)
        high_conf = 0.75
        mid_conf = 0.60

        if confidence_score >= high_conf:
            verdict = 'Likely Fake' if int(p) == 1 else 'Likely Real'
        elif confidence_score >= mid_conf:
            verdict = 'Possibly Fake' if int(p) == 1 else 'Possibly Real'
        else:
            verdict = 'Uncertain'

        explanation = f"Model predicts '{label_str}' with {confidence_pct}% confidence (fake_probability={prob:.3f})."

        results.append({
            'label': label_str,
            'is_fake': bool(int(p)),
            'fake_probability': float(prob),
            'confidence_score': round(confidence_score, 4),
            'confidence': confidence_pct,
            'verdict': verdict,
            'explanation': explanation,
        })
    return results
