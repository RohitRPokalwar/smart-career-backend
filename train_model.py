import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

# Step 1: Load CSV
data_path = "./data/career_dataset.csv"
df = pd.read_csv(data_path)

# Step 2: Features and Labels
X = df["skills"]
y = df["domain"]

# Step 3: Vectorize the skill text
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Step 4: Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Step 5: Save model and vectorizer
os.makedirs("../model", exist_ok=True)

with open("../model/career_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
