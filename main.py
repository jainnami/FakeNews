import re
import numpy as np
import pandas as pd
import argparse
import os
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import MultinomialNB


# -- Load data (files have header row) ---------------------------------
train_df = pd.read_csv("train.tsv", sep='\t', header=0)
valid_df = pd.read_csv("valid.tsv", sep='\t', header=0)
test_df = pd.read_csv("test.tsv", sep='\t', header=0)

# Command-line args
parser = argparse.ArgumentParser()
parser.add_argument('--skip-transformer', action='store_true', help='Skip transformer embedding step')
parser.add_argument('--cache-embeddings', action='store_true', help='Save/load transformer embeddings to/from disk')
parser.add_argument('--embeddings-file', default='embeddings.npz', help='File path for cached embeddings')
args = parser.parse_args()


# -- Utilities -----------------------------------------------------------
def normalize_label(l):
    if pd.isna(l):
        return None
    s = str(l).strip().lower()
    true_set = {"true", "mostly-true", "mostly true", "half-true", "half true"}
    false_set = {"barely-true", "barely true", "false", "pants-fire", "pants on fire", "pants-on-fire"}
    if s in true_set:
        return 'true'
    if s in false_set:
        return 'false'
    if 'true' in s and 'barely' not in s and 'false' not in s and 'pants' not in s:
        return 'true'
    if 'false' in s or 'pants' in s or 'barely' in s:
        return 'false'
    return None


def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\S+', ' ', s)
    s = re.sub(r"\"+", ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# -- Prepare labels and new text features --------------------------------
for df in (train_df, valid_df, test_df):
    df['label_bin'] = df['label'].apply(normalize_label)
    df['clean_statement'] = df['statement'].apply(clean_text)
    df['statement_len'] = df['clean_statement'].apply(lambda x: len(x.split()))

# Drop rows where label is None (unmapped)
train_df = train_df[train_df['label_bin'].notna()].reset_index(drop=True)
valid_df = valid_df[valid_df['label_bin'].notna()].reset_index(drop=True)
test_df = test_df[test_df['label_bin'].notna()].reset_index(drop=True)


# -- Feature transforms --------------------------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=15000, ngram_range=(1,3))
ohe = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

def build_feature_matrices(train_df, valid_df, test_df, fit=True):
    # Text
    X_train_text = vectorizer.fit_transform(train_df['clean_statement']) if fit else vectorizer.transform(train_df['clean_statement'])
    X_valid_text = vectorizer.transform(valid_df['clean_statement'])
    X_test_text = vectorizer.transform(test_df['clean_statement'])

    # Categorical: include more metadata (party, state, speaker, job title, context)
    cat_cols = ['party', 'state', 'speaker', 'speaker_job_title', 'context']
    train_cat = train_df[cat_cols].fillna('missing')
    valid_cat = valid_df[cat_cols].fillna('missing')
    test_cat = test_df[cat_cols].fillna('missing')

    if fit:
        X_train_cat = ohe.fit_transform(train_cat)
        X_valid_cat = ohe.transform(valid_cat)
        X_test_cat = ohe.transform(test_cat)
    else:
        X_train_cat = ohe.transform(train_cat)
        X_valid_cat = ohe.transform(valid_cat)
        X_test_cat = ohe.transform(test_cat)

    # Numeric
    train_num = train_df[['statement_len']].fillna(0).astype(float)
    valid_num = valid_df[['statement_len']].fillna(0).astype(float)
    test_num = test_df[['statement_len']].fillna(0).astype(float)
    if fit:
        train_num_scaled = scaler.fit_transform(train_num)
    else:
        train_num_scaled = scaler.transform(train_num)
    valid_num_scaled = scaler.transform(valid_num)
    test_num_scaled = scaler.transform(test_num)

    # Convert numeric to sparse
    train_num_sparse = sparse.csr_matrix(train_num_scaled)
    valid_num_sparse = sparse.csr_matrix(valid_num_scaled)
    test_num_sparse = sparse.csr_matrix(test_num_scaled)

    # Combine all sparse matrices horizontally
    X_train = sparse.hstack([X_train_text, X_train_cat, train_num_sparse], format='csr')
    X_valid = sparse.hstack([X_valid_text, X_valid_cat, valid_num_sparse], format='csr')
    X_test = sparse.hstack([X_test_text, X_test_cat, test_num_sparse], format='csr')

    return X_train, X_valid, X_test


# Build features
X_train, X_valid, X_test = build_feature_matrices(train_df, valid_df, test_df, fit=True)
y_train = train_df['label_bin'].values
y_valid = valid_df['label_bin'].values
y_test = test_df['label_bin'].values


# -- Optional balancing: try to import imbalanced-learn's RandomOverSampler
oversampler = None
try:
    from imblearn.over_sampling import RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
except Exception:
    oversampler = None

if oversampler is not None:
    try:
        X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)
        X_train, y_train = X_train_res, y_train_res
    except Exception:
        pass


# -- Model and tuning ----------------------------------------------------
clf = LogisticRegression(max_iter=2000, solver='saga', penalty='l2', class_weight='balanced')
param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(clf, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
gs.fit(X_train, y_train)

best = gs.best_estimator_
print('Best params:', gs.best_params_)


# Evaluate on validation and test sets
def evaluate(model, X, y, split_name='test'):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n{split_name.title()} Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))
    # ROC AUC for binary
    try:
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score([1 if yy=='true' else 0 for yy in y], probs)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
        pass

evaluate(best, X_valid, y_valid, 'validation')
evaluate(best, X_test, y_test, 'test')

# --- Train and evaluate MultinomialNB on non-negative features (raw numeric)
# Build NB feature matrices using same vectorizer and ohe but raw numeric (no StandardScaler)
cat_cols = ['party', 'state', 'speaker', 'speaker_job_title', 'context']
train_cat = train_df[cat_cols].fillna('missing')
valid_cat = valid_df[cat_cols].fillna('missing')
test_cat = test_df[cat_cols].fillna('missing')

X_train_text = vectorizer.transform(train_df['clean_statement'])
X_valid_text = vectorizer.transform(valid_df['clean_statement'])
X_test_text = vectorizer.transform(test_df['clean_statement'])

X_train_cat = ohe.transform(train_cat)
X_valid_cat = ohe.transform(valid_cat)
X_test_cat = ohe.transform(test_cat)

train_num_raw = train_df[['statement_len']].fillna(0).astype(float)
valid_num_raw = valid_df[['statement_len']].fillna(0).astype(float)
test_num_raw = test_df[['statement_len']].fillna(0).astype(float)

train_num_sparse_raw = sparse.csr_matrix(train_num_raw.values)
valid_num_sparse_raw = sparse.csr_matrix(valid_num_raw.values)
test_num_sparse_raw = sparse.csr_matrix(test_num_raw.values)

X_train_nb = sparse.hstack([X_train_text, X_train_cat, train_num_sparse_raw], format='csr')
X_valid_nb = sparse.hstack([X_valid_text, X_valid_cat, valid_num_sparse_raw], format='csr')
X_test_nb = sparse.hstack([X_test_text, X_test_cat, test_num_sparse_raw], format='csr')

# If oversampling available, apply to NB training set as well
if oversampler is not None:
    try:
        X_train_nb, y_train_nb = oversampler.fit_resample(X_train_nb, y_train)
    except Exception:
        y_train_nb = y_train
else:
    y_train_nb = y_train

nb = MultinomialNB()
nb.fit(X_train_nb, y_train_nb)

print('\n--- MultinomialNB Results ---')
evaluate(nb, X_valid_nb, y_valid, 'validation (NB)')
evaluate(nb, X_test_nb, y_test, 'test (NB)')

print('\n--- LogisticRegression Results (best) ---')
evaluate(best, X_valid, y_valid, 'validation (LR)')
evaluate(best, X_test, y_test, 'test (LR)')

print('\nComparison:')
def simple_metrics(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    try:
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score([1 if yy=='true' else 0 for yy in y], probs)
    except Exception:
        auc = None
    return acc, auc

acc_nb, auc_nb = simple_metrics(nb, X_test_nb, y_test)
acc_lr, auc_lr = simple_metrics(best, X_test, y_test)
print(f"Test acc - NB: {acc_nb:.4f}, LR: {acc_lr:.4f}")
if auc_nb is not None and auc_lr is not None:
    print(f"Test AUC - NB: {auc_nb:.4f}, LR: {auc_lr:.4f}")

print('\nAttempting transformer-based embeddings (this may be slow)...')
def try_transformer_embeddings(args, model_name='distilbert-base-uncased', batch_size=32):
    # If user requested cached embeddings and file exists, load and return
    if args.cache_embeddings and os.path.exists(args.embeddings_file):
        print(f'Loading cached embeddings from {args.embeddings_file}')
        d = np.load(args.embeddings_file)
        emb_train = d['train']
        emb_valid = d['valid']
        emb_test = d['test']
        cat_cols = ['party', 'state', 'speaker', 'speaker_job_title', 'context']
        train_cat = train_df[cat_cols].fillna('missing')
        valid_cat = valid_df[cat_cols].fillna('missing')
        test_cat = test_df[cat_cols].fillna('missing')
        emb_cat_train = ohe.transform(train_cat)
        emb_cat_valid = ohe.transform(valid_cat)
        emb_cat_test = ohe.transform(test_cat)
        train_num = train_df[['statement_len']].fillna(0).astype(float).values
        valid_num = valid_df[['statement_len']].fillna(0).astype(float).values
        test_num = test_df[['statement_len']].fillna(0).astype(float).values
        emb_train_sparse = sparse.csr_matrix(emb_train)
        emb_valid_sparse = sparse.csr_matrix(emb_valid)
        emb_test_sparse = sparse.csr_matrix(emb_test)
        X_tr = sparse.hstack([emb_train_sparse, emb_cat_train, sparse.csr_matrix(train_num)], format='csr')
        X_va = sparse.hstack([emb_valid_sparse, emb_cat_valid, sparse.csr_matrix(valid_num)], format='csr')
        X_te = sparse.hstack([emb_test_sparse, emb_cat_test, sparse.csr_matrix(test_num)], format='csr')
        return (X_tr, X_va, X_te)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception:
        print('Transformers or torch not installed — skipping transformer experiments. To enable, pip install transformers torch')
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Building transformer embeddings on device: {device}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def embed_texts(texts):
        embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                toks = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=256)
                toks = {k: v.to(device) for k, v in toks.items()}
                out = model(**toks)
                last = out.last_hidden_state
                mask = toks['attention_mask'].unsqueeze(-1)
                summed = (last * mask).sum(1)
                denom = mask.sum(1).clamp(min=1)
                pooled = (summed / denom).cpu().numpy()
                embeds.append(pooled)
        return np.vstack(embeds)

    train_texts = train_df['clean_statement'].astype(str).tolist()
    valid_texts = valid_df['clean_statement'].astype(str).tolist()
    test_texts = test_df['clean_statement'].astype(str).tolist()

    emb_train = embed_texts(train_texts)
    emb_valid = embed_texts(valid_texts)
    emb_test = embed_texts(test_texts)

    # Save embeddings if requested
    if args.cache_embeddings:
        print(f'Saving embeddings to {args.embeddings_file} (this may take space)')
        np.savez_compressed(args.embeddings_file, train=emb_train, valid=emb_valid, test=emb_test)

    cat_cols = ['party', 'state', 'speaker', 'speaker_job_title', 'context']
    train_cat = train_df[cat_cols].fillna('missing')
    valid_cat = valid_df[cat_cols].fillna('missing')
    test_cat = test_df[cat_cols].fillna('missing')
    emb_cat_train = ohe.transform(train_cat)
    emb_cat_valid = ohe.transform(valid_cat)
    emb_cat_test = ohe.transform(test_cat)

    train_num = train_df[['statement_len']].fillna(0).astype(float).values
    valid_num = valid_df[['statement_len']].fillna(0).astype(float).values
    test_num = test_df[['statement_len']].fillna(0).astype(float).values

    emb_train_sparse = sparse.csr_matrix(emb_train)
    emb_valid_sparse = sparse.csr_matrix(emb_valid)
    emb_test_sparse = sparse.csr_matrix(emb_test)

    X_tr = sparse.hstack([emb_train_sparse, emb_cat_train, sparse.csr_matrix(train_num)], format='csr')
    X_va = sparse.hstack([emb_valid_sparse, emb_cat_valid, sparse.csr_matrix(valid_num)], format='csr')
    X_te = sparse.hstack([emb_test_sparse, emb_cat_test, sparse.csr_matrix(test_num)], format='csr')

    return (X_tr, X_va, X_te)

tm = try_transformer_embeddings(args)
if tm is not None:
    X_tr_emb, X_va_emb, X_te_emb = tm
    emb_clf = LogisticRegression(max_iter=2000, solver='saga', class_weight='balanced')
    emb_clf.fit(X_tr_emb, y_train)
    print('\n--- Transformer (LR) Results ---')
    evaluate(emb_clf, X_va_emb, y_valid, 'validation (Transformer)')
    evaluate(emb_clf, X_te_emb, y_test, 'test (Transformer)')
else:
    print('Transformer path skipped.')

print('\nDone')