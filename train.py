import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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

print(f'Cleaned dataset ready for vectorization.Total rows: {len(df)}')

X = df['combined_text']
y = df['fraudulent']

print('Splitting data...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Vectorizing Text (limiting to top 5000 features)...')
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)

print('Training Random Forest model (this may take a minute)...')

model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
)
model.fit(X_train_tfidf, y_train)

#quick validation
X_test_tfidf = tfidf.transform(X_test)
train_score = model.score(X_train_tfidf, y_train)
test_score = model.score(X_test_tfidf, y_test)
print(f'Model Training Accuracy: {train_score:.4f}')
print(f'Model Test Accuracy: {test_score:.4f}')

print('Saving Model and Vectorizer')
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print('Training complete! Files saved successfully.')