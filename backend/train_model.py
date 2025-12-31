import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def discover_text_column(df):
    for name in ['text', 'content', 'article', 'body']:
        if name in df.columns:
            return name
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError('No text column found in dataset')


def discover_label_column(df):
    # prefer explicit label-like names
    for name in ['label', 'target', 'class']:
        if name in df.columns:
            return name

    # prefer columns that look binary / low-cardinality (likely label)
    for c in df.columns[::-1]:
        vals = df[c].dropna().unique()
        if len(vals) == 0:
            continue
        if len(vals) <= 2:
            return c

    # fallback: last column
    return df.columns[-1]


def normalize_labels(y):
    y = y.copy()
    mapping = {
        'fake': 1, 'FAKE': 1, 'Fake': 1,
        'real': 0, 'REAL': 0, 'Real': 0,
        'true': 0, 'True': 0, 'TRUE': 0,
        'false': 0, 'False': 0, 'FALSE': 0,
        '1': 1, '0': 0
    }
    y = y.replace(mapping)

    # try numeric conversion
    numeric = pd.to_numeric(y, errors='coerce')
    if not numeric.isna().all():
        if numeric.isna().any():
            # some values failed conversion; fall through to next attempt
            pass
        else:
            return numeric.astype(int)

    # final attempt: astype int
    try:
        return y.astype(int)
    except Exception:
        unique_vals = pd.Series(y.dropna().unique()).astype(str)
        sample = unique_vals[:20].tolist()
        raise ValueError(
            "Could not normalize label column to integer labels. "
            f"Sample unique values: {sample}.\n"
            "If your dataset's label column is not the last column, pass it with --label. "
            "Alternatively, provide a CSV with a binary label column (0/1 or 'fake'/'real')."
        )


def train(dataset_path, model_out=None, label_name=None, create_label=False):
    df = pd.read_csv(dataset_path)
    text_col = discover_text_column(df)
    label_col = label_name if label_name is not None else discover_label_column(df)
    print(f"Using text column '{text_col}' and label column '{label_col}'")

    # interactive fallback for non-binary auto-detection
    if label_name is None and label_col in df.columns:
        uniq = df[label_col].dropna().unique()
        if len(uniq) > 2:
            print(f"Label column '{label_col}' has {len(uniq)} unique values (sample: {list(map(str, uniq[:10]))}).")
            print("If this is incorrect, re-run with --label COLUMN. Use --create-label to add a default 'label' column.")

    X = df[text_col].fillna('').astype(str)

    # attempt to normalize labels; handle failures gracefully
    try:
        y = normalize_labels(df[label_col])
    except Exception as exc:
        if create_label:
            print("Could not normalize existing label column; creating default 'label' column with 0 values.")
            df['label'] = 0
            label_col = 'label'
            y = df['label'].astype(int)
        else:
            print(str(exc))
            raise SystemExit(2)

    # need at least two classes to train
    if y.nunique() < 2:
        print(f"Label column '{label_col}' has only one class ({y.unique()}). Training requires at least two classes.")
        print("Provide a labeled dataset with two classes or re-run with --label pointing to a binary column.")
        raise SystemExit(2)

    strat = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', token_pattern=r"\b[a-z]{3,}\b")),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    print('Training model...')
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1]
    print('\nTest Classification Report:')
    print(classification_report(y_test, preds))
    try:
        print('ROC-AUC:', roc_auc_score(y_test, probs))
    except Exception:
        pass

    # default model path inside backend directory if not provided
    if model_out is None:
        model_out = os.path.join(os.path.dirname(__file__), 'model.pkl')
    os.makedirs(os.path.dirname(model_out) or '.', exist_ok=True)
    joblib.dump(pipeline, model_out)
    print(f'Model saved to {model_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?', default=os.path.join(os.path.dirname(__file__), 'fake_news_dataset.csv'), help='Path to CSV dataset')
    parser.add_argument('--out', default=None, help='Output model path')
    parser.add_argument('--label', default=None, help='Name of the label column (overrides auto-detection)')
    parser.add_argument('--create-label', action='store_true', help="If set, create a default 'label' column with 0 values when normalization fails")
    args = parser.parse_args()

    # verify dataset exists; allow a sensible fallback and give a clear error
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # if default expected name isn't present, try the older filename
        alt = os.path.join(os.path.dirname(__file__), 'fake_review_dataset.csv')
        if (dataset_path == os.path.join(os.path.dirname(__file__), 'fake_news_dataset.csv')
                and os.path.exists(alt)):
            print(f"Default dataset not found; using alternate dataset: {alt}")
            dataset_path = alt
        else:
            print(f"Error: dataset not found: {dataset_path}\nProvide a valid CSV path or place 'fake_news_dataset.csv' in the backend directory.")
            raise SystemExit(2)

    train(dataset_path, args.out, label_name=args.label, create_label=args.create_label)
