import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numba import njit, prange
import re
from collections import Counter

# -------------------------------
# Config
# -------------------------------
max_vocab_size = 5000  # Limit the vocabulary to top 5000 words

# -------------------------------
# Data Preprocessing
# -------------------------------

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_vocab(texts, max_size=None):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    most_common = counter.most_common(max_size)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    return vocab

def vectorize_texts(texts, vocab):
    vectors = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in tokenize(text):
            if word in vocab:
                vectors[i, vocab[word]] += 1
    return vectors

# -------------------------------
# Logistic Regression from Scratch
# -------------------------------

@njit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@njit(parallel=True)
def predict(X, weights):
    m, n = X.shape
    predictions = np.zeros(m)
    for i in prange(m):
        z = 0
        for j in range(n):
            z += X[i, j] * weights[j]
        predictions[i] = sigmoid(z)
    return predictions

@njit(parallel=True)
def compute_gradient(X, y, predictions):
    m, n = X.shape
    gradient = np.zeros(n)
    for j in prange(n):
        for i in range(m):
            gradient[j] += (predictions[i] - y[i]) * X[i, j]
        gradient[j] /= m
    return gradient

@njit
def update_weights(weights, gradient, lr):
    for i in range(len(weights)):
        weights[i] -= lr * gradient[i]
    return weights

def train(X, y, lr=0.01, epochs=100):
    weights = np.zeros(X.shape[1], dtype=np.float32)
    for epoch in range(epochs):
        predictions = predict(X, weights)
        gradient = compute_gradient(X, y, predictions)
        weights = update_weights(weights, gradient, lr)

        if epoch % 10 == 0:
            loss = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
    return weights

# -------------------------------
# Accuracy
# -------------------------------

def evaluate(predictions, y_true):
    y_pred = (predictions >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_true)
    return accuracy

# -------------------------------
# Example Usage
# -------------------------------

# Load dataset (CSV with "review","sentiment" format)
df = pd.read_csv("movie_reviews.csv")
print("Available columns:", df.columns.tolist())

texts = df['review'].astype(str).tolist()
labels = df['sentiment'].map({'positive': 1, 'negative': 0}).astype(int).values

# Build vocabulary and vectorize
vocab = build_vocab(texts, max_size=max_vocab_size)
X = vectorize_texts(texts, vocab)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
weights = train(X_train, y_train, lr=0.01, epochs=100)

# Predict & evaluate
preds = predict(X_test, weights)
acc = evaluate(preds, y_test)
print(f"\n✅ Final Accuracy on Test Set: {acc * 100:.2f}%")
