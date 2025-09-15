# train_model.py (Updated to use FakeNewsNet.csv directly)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Starting the model training process...")

# 1. Load the dataset
try:
    # This line is now correctly pointing to your file
    df = pd.read_csv('FakeNewsNet.csv') 
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: FakeNewsNet.csv not found. Make sure the file is in your project folder.")
    exit()

# 2. Preprocess the data
# This dataset uses 'title' for the text and 'real' for the label.
df.dropna(subset=['title', 'real'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Separate features (X) and labels (y)
X = df['title']
# The label 'real' is 1 for real news and 0 for fake. We flip it to match our logic.
# Our model will predict 1 for FAKE, 0 for REAL.
y = 1 - df['real']

print(f"Dataset shape: {df.shape}")

# 3. Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)
print("Text has been vectorized.")

# 4. Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = PassiveAggressiveClassifier(max_iter=50)
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluate and save the trained model and vectorizer
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
    
print("Model and vectorizer have been saved successfully.")