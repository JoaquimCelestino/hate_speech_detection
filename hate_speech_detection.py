
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Paths to the data files
train_file = r'c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\Text data\datatext\train_E6oV3lV.csv'
test_file = r'c:\Users\JOAQUIM CELESTINO\Downloads\DataSCIENCEPACK\Text data\datatext\test_tweets_anuFYb8.csv'

# Load training data
print("Loading training data...")
train_data = pd.read_csv(train_file)

# Display a sample of the data
print("Sample of training data:")
print(train_data.head())

# Preprocessing: Removing punctuation, converting to lowercase, etc.
def preprocess_text(text):
    """Function to clean and preprocess text data."""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

train_data['tweet'] = train_data['tweet'].apply(preprocess_text)

# Splitting the training data into features and labels
X = train_data['tweet']
y = train_data['label']

# Vectorizing text data using TF-IDF
print("Vectorizing text data...")
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 1))
X_tfidf = tfidf.fit_transform(X)

# Splitting into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)

# Model training using Logistic Regression
print("Training the model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate on the validation set
y_pred_val = model.predict(X_val)
val_f1 = f1_score(y_val, y_pred_val)
print(f"Validation F1 Score: {val_f1:.4f}")

# Load test data
print("Loading test data...")
test_data = pd.read_csv(test_file)
test_data['tweet'] = test_data['tweet'].apply(preprocess_text)
X_test = tfidf.transform(test_data['tweet'])

# Predicting on test data
print("Predicting on test data...")
y_test_pred = model.predict(X_test)

# Saving predictions to a CSV file
output_file = 'test_predictions.csv'
print(f"Saving predictions to {output_file}...")
np.savetxt(output_file, y_test_pred, fmt='%d')

print("Task complete. Predictions saved.")
    
