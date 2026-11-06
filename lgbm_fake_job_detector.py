import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, precision_recall_curve, auc, average_precision_score)

print('Loading and Validating Dataset...')
df = pd.read_csv('fake_job_postings.csv')

print(f'Total Rows & Columns: {df.shape}')
print('Missing values per column:', df.isnull().sum())

print(
    'Class Distribution (0 = Real, 1 = Fake):\n',
    df['fraudulent'].value_counts(normalize=True) * 100,
)

print('Data Cleaning...')
initial_rows = len(df)
df = df.drop_duplicates()
print(f'Removed {initial_rows - len(df)} duplicate rows.')

text_columns = [
    'title',
    'company_profile',
    'description',
    'requirements',
    'benefits',
]
for col in text_columns:
    df[col] =  df[col].fillna('')

df['combined_text'] = (
    df['title']
    + ' '
    + df['company_profile']
    + ' '
    + df['requirements']
    + ' '
    + df['benefits']
)

print(f'Cleaned dataset.Total rows: {len(df)}')

#STRUCTURAL METADATA
metadata_columns = ['has_company_logo', 'telecommuting', 'has_questions']
for col in metadata_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)
    else:
        df[col] = 0

#caching embeddings
embeddings_file = 'text_embeddings_minilm.npy'
embedder = None

if os.path.exists(embeddings_file):
    print('Loading pre-computed Ytansformer Embeddings from disk')
    text_embeddings = np.load(embeddings_file)
else:
    print('Generating Transformer Embeddings (all-MiniLM-L6-v2)')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = embedder.encode(df['combined_text'].tolist(), show_progress_bar=True)
    np.save(embeddings_file, text_embeddings)
    print('Embeddings generated and saved succesfully!')
    
print(f'Text Embeddings Shape: {text_embeddings.shape}')
 
print('Extracting and Stacking Structural Metadata...')
metadata_features = df[metadata_columns].values

X_combined = np.hstack((text_embeddings, metadata_features))
y = df['fraudulent'].values
print(f'Combined Multi-Modal Matrix Shape: {X_combined.shape}')

#5-FOLD STRATIFIED CROSS-VALIDATION
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_scores = []
ap_scores = []

oof_probabilities = np.zeros(len(y))
oof_true = np.zeros(len(y))

print('\n Starting 5-fold Cross-Validation...')
for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, y)):
    X_train, X_test = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    lgbm_model = LGBMClassifier(
        n_estimators=100, learning_rate = 0.05, scale_pos_weight = scale_weight, random_state=42, n_jobs=-1
    )
    lgbm_model.fit(X_train, y_train)

    y_proba = lgbm_model.predict_proba(X_test)[:, 1]
    oof_probabilities[test_idx] = y_proba
    oof_true[test_idx] = y_test

    fold_roc = roc_auc_score(y_test, y_proba)
    fold_ap = average_precision_score(y_test, y_proba)

    roc_auc_scores.append(fold_roc)
    ap_scores.append(fold_ap)

    print(f'Fold {fold + 1} | ROAC-AUC: {fold_roc:.4f} | PR-AUC: {fold_ap:.4f}')

print('CROSS VALIDATION FINAL SUMMARY')
print(f'Mean ROC-AUC: {np.mean(roc_auc_scores):.4f} ± {np.std(roc_auc_scores):.4f}')
print(f'Mean PR-AUC: {np.mean(ap_scores):.4f} ± {np.std(ap_scores):.4f}')


#PRECISION-RECALL CURVE
precision, recall, thresholds = precision_recall_curve(oof_true, oof_probabilities)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10,8))
plt.plot(recall, precision, color='orange', lw=2, label=f'Out-of-Fold PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall (Sensitivity)', fontsize=14)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=14)
plt.title('PRECISION-RECALL CURVE (Multi-Modal LightGBM)', fontsize=18, fontweight='bold')
plt.legend(loc='lower left', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('precision_recall_curve.png', dpi=400, bbox_inches='tight')
print('\nPR Curve plot saved successfully!')
plt.show()


#FINAL MODEL FULL DATASET TRAINING
print('Training final model on full dataset...')
neg_total = np.sum(y == 0)
pos_total = np.sum(y == 1)
final_scale_weight = neg_total / pos_total if pos_total > 0 else 1.0

print('Training LightGBM model with sequential residual boosting...')
final_lgbm_model = LGBMClassifier(
    n_estimators=100, learning_rate = 0.05, scale_pos_weight = final_scale_weight, random_state=42, n_jobs=-1
)
final_lgbm_model.fit(X_combined, y)


print('Saving Proposed Model and Embedder...')
joblib.dump(final_lgbm_model, 'fake_job_lgbm_model.pkl')
if embedder is not None:
    joblib.dump(embedder, 'sentence_embedder.pkl')
print('Training complete! Files saved successfully.')