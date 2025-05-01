import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load the datasets
fake = pd.read_csv(r"D:\Fake News\data\fake.csv")
true = pd.read_csv(r"D:\Fake News\data\true.csv")


# Add labels (0 for fake, 1 for true)
fake['label'] = 0
true['label'] = 1

# Combine fake and true data
data = pd.concat([fake, true])

# Use headlines (titles) instead of full articles
data['text'] = data['title']  # This makes the model learn from short text

# Clean the text data
def clean(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())  # Remove non-alphanumeric characters
    return text.strip()

data['text'] = data['text'].apply(clean)

# Split data into features (X) and labels (y)
X = data['text']
y = data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=2, ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Save the model and vectorizer for later use
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved.")

