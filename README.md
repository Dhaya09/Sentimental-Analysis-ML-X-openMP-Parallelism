# 🚀 Optimized Sentiment Analysis using Logistic Regression + Numba (OpenMP)

A high-performance implementation of a sentiment analysis model using **Numba**-accelerated **logistic regression** for fast matrix operations. This project demonstrates real-time classification of movie reviews into **positive** or **negative** sentiments — with drastically reduced training time thanks to **OpenMP-style parallelism** using `@njit(parallel=True)`.

---

## 🧠 Project Summary

- **Goal:** Speed up sentiment analysis by optimizing matrix operations in logistic regression.
- **Technique:** Bag-of-words vectorization + logistic regression (custom implemented).
- **Optimization:** Leveraged [Numba](https://numba.pydata.org/) and OpenMP-style parallelization (`prange`) for:
  - Prediction vector dot products
  - Gradient descent computation

---

## 📁 Dataset

- Source : [DataSet](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Input file: `movie_reviews.csv`
- Required columns:
  - `review`: Free-form movie review text
  - `sentiment`: Either `positive` or `negative`

---

## 🔧 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

### 📦 requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.0
numba>=0.53.0
```

---

## 🧪 How to Run

```bash
python sentiment_analysis.py
python sentiment_analysis_openMP.py
```

What it does:
- Loads and preprocesses the dataset
- Builds a vocabulary and vectorizes text
- Trains a logistic regression model using gradient descent
- Evaluates accuracy and prints training time
- Enters **interactive live prediction mode**

---

## 💬 Live Sentiment Prediction Demo

After training, enter your own review:

```
Enter a movie review (or type 'exit' to quit):
> this movie was amazing and inspiring

Your Review: "this movie was amazing and inspiring"
Predicted Sentiment: Positive 😊
```

---

## 📊 Performance Summary

| Setup                          | Training Time | Accuracy |
|-------------------------------|---------------|----------|
| Without Numba/OpenMP (100 epochs) | ~120 sec      | ~75%     |
| With Numba + OpenMP (100 epochs) | ~60 sec       | ~75%     |
| With Numba + OpenMP (50 epochs)  | ~30 sec       | ~70%     |

✅ Accuracy is retained while reducing compute time by ~2x.

---

## 🧠 Techniques Used

- Logistic Regression from scratch (no sklearn model)
- Gradient Descent with sigmoid activation
- Bag-of-Words Vectorization
- Parallelization with Numba + OpenMP
- Live user input prediction via terminal

---

## 🚀 Future Enhancements

- Add TF-IDF vectorization
- Save and reload trained model (pickle or NumPy)
- Build GUI using Flask or Tkinter
- Add support for more granular sentiment labels

---

## 👨‍💻 Team Members

- **Dhayanidhi S** – 23BIT0214  
- **Harisankaran M** – 23BIT0150  
- **Prashaanth Raj** – 23BIT0173

---

## 📚 References

1. Pang, B., & Lee, L. (2008). *Opinion mining and sentiment analysis*.
2. Refaeilzadeh, P., Tang, L., & Liu, H. (2009). *Cross-validation*.
3. Numba.org – *High-Performance Python with LLVM*.
4. *Parallelization of Logistic Regression Using OpenMP* – International Journal of Computer Applications.

---

## 📄 License
feel free to use, modify, and share!
