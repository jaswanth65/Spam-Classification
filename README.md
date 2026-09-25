# 📩 SMS Spam Classifier — Naive Bayes

A complete, from-scratch pipeline for classifying SMS messages as **Spam** or **Ham (not spam)** using classical NLP preprocessing, Bag-of-Words feature extraction, and a **Multinomial Naive Bayes** classifier — tuned with `GridSearchCV` and evaluated with confusion matrix, ROC, and Precision-Recall curves.

This repo walks through the full journey: raw text → cleaned tokens → numerical features → trained model → evaluated, saved, production-ready classifier.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Feature Extraction](#-feature-extraction)
- [Model Training & Tuning](#-model-training--tuning)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

Spam clutters inboxes and enables phishing attacks, so filtering it reliably matters. This project builds a spam detector using **Naive Bayes**, an algorithm that's a natural fit for text classification because it's fast, works well on sparse high-dimensional data, and scales cleanly to large vocabularies.

Instead of trying to define what a spam message "looks like," the model learns the **historical frequency of words** in spam vs. ham messages, then uses Bayes' Theorem to compute the probability that a *new* message is spam, given the words it contains.

---

## 📊 Dataset

The project uses the classic **SMS Spam Collection Dataset**, a tab-separated file of 2 columns:

| Column    | Description                          |
|-----------|---------------------------------------|
| `label`   | `ham` or `spam`                       |
| `message` | The raw SMS text                      |

**Key stats:**

| Stage                  | Total Messages | Ham   | Spam |
|-------------------------|:--------------:|:-----:|:----:|
| Raw dataset              | 5,572          | 4,825 | 747  |
| After removing 403 duplicates | 5,169    | 4,516 | 653  |

The classes are **imbalanced** — roughly **87% ham vs. 13% spam** — which matters a lot for how the model is evaluated (see [Results](#-results)).

---

## 🧠 How It Works

### Bayes' Theorem

```
P(Spam | Words) = [ P(Words | Spam) * P(Spam) ] / P(Words)
```

| Term | Meaning |
|---|---|
| `P(Spam)` | **Prior** — how likely *any* message is spam, before reading it |
| `P(Words \| Spam)` | **Likelihood** — how likely these words are, given the message is spam |
| `P(Spam \| Words)` | **Posterior** — the final verdict, spam probability given the words |

### The "Naive" Assumption

Naive Bayes assumes every word is **independent** of every other word — grammar and word order are ignored entirely. To score a message like `"send money now"`, it just multiplies the individual word probabilities:

```
P("send money now" | Spam) ≈ P("send"|Spam) × P("money"|Spam) × P("now"|Spam)
```

This is "naive" because in reality words *aren't* independent — but the approximation works remarkably well for text classification and is extremely cheap to compute.

---

## 🧹 Preprocessing Pipeline

Every raw message is normalized identically before it reaches the model (see `preprocess_message()` in the notebook):

1. **Lowercase** — `"Free"` and `"free"` become the same token.
2. **Strip punctuation & numbers** — keep only lowercase letters, whitespace, `$`, and `!` (these two symbols carry spam signal — money amounts and urgency/emphasis).
3. **Tokenize** — split into words using NLTK's `punkt` tokenizer (`word_tokenize`), which correctly handles abbreviations and edge cases a naive `.split()` would break on.
4. **Remove stop words** — drop common low-signal words (`the`, `is`, `and`, ...) using NLTK's English stopword list.
5. **Stem** — reduce words to their root form with `PorterStemmer` (`running` → `run`), shrinking the vocabulary and collapsing word variants.
6. **Rejoin tokens** into a single space-separated string, ready for vectorization.

> ⚠️ **Critical:** the exact same `preprocess_message()` function must be applied to any new message at inference time — training and inference preprocessing must always match.

---

## 🔠 Feature Extraction

Text is converted into numeric vectors using **`CountVectorizer`** (Bag-of-Words):

```python
vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
```

| Parameter | Purpose |
|---|---|
| `min_df=1` | Keep a word even if it appears in only 1 message (no rare-word filtering here) |
| `max_df=0.9` | Drop words appearing in >90% of messages — too common to be discriminating |
| `ngram_range=(1, 2)` | Include both **unigrams** (`free`) and **bigrams** (`free prize`) — bigrams capture short-range word order that pure unigrams lose |

**Result on this dataset:** a vocabulary of **37,069 unique unigrams + bigrams** across **5,169 messages**, producing a sparse matrix with **80,181** non-zero entries (most of each message's 37,069-dimension vector is zero).

---

## 🤖 Model Training & Tuning

### Why a `Pipeline`?

`CountVectorizer` and `MultinomialNB` are locked together into a single `Pipeline` object. This solves a **dimension-mismatch problem**: a lone new message vectorized on its own would only produce a handful of dimensions, not the 37,069 the trained model expects. The pipeline remembers the original vocabulary and always projects new text into that same fixed-size vector space.

```python
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])
```

### Hyperparameter Tuning — `alpha` (Laplace Smoothing)

Naive Bayes multiplies word probabilities together. If a word (e.g. `"invoice"`) **never** appeared in spam during training, its probability is `0` — and that single zero collapses the *entire* product to zero, even if every other word screamed "spam." `alpha` prevents this **zero-frequency problem** by adding a small smoothing count to every word, so no probability is ever exactly zero.

```python
param_grid = {"classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
grid_search.fit(df["message"], y)
```

**Best result:** `alpha = 0.25`, selected via 5-fold cross-validation on **F1-score** (a balanced measure of precision and recall — important given the class imbalance).

---

## 📈 Results

Evaluated on a held-out **20% stratified test split** (1,034 messages: 903 ham, 131 spam):

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| Ham (0) | 0.9845 | 0.9878 | 0.9862 | 903 |
| Spam (1) | 0.9141 | 0.8931 | 0.9035 | 131 |
| **Accuracy** | | | **0.9758** | 1034 |
| Macro avg | 0.9493 | 0.9405 | 0.9448 | 1034 |
| Weighted avg | 0.9756 | 0.9758 | 0.9757 | 1034 |

**Confusion Matrix:**

|  | Predicted Ham | Predicted Spam |
|---|:---:|:---:|
| **Actual Ham**  | 892 (TN) | 11 (FP) |
| **Actual Spam** | 14 (FN)  | 117 (TP) |

**Curve metrics:**

- **ROC-AUC:** `0.9744`
- **PR-AUC:** `0.9471`

> **Why both curves?** With ~87% ham / 13% spam, the ROC curve's False Positive Rate denominator (`FP + TN`) is dominated by the huge number of true negatives, which can make ROC look artificially strong. The **Precision-Recall curve ignores true negatives entirely** and focuses on how well the model isolates the rare spam class without burying legitimate messages — making it the more trustworthy metric here.

### Example Predictions

| Message (truncated) | Prediction | Spam Prob. |
|---|:---:|:---:|
| "Congratulations! You've won a $1000 Walmart gift card..." | Spam | 1.00 |
| "Hey, are we still meeting up for lunch today?" | Not-Spam | 0.00 |
| "Urgent! Your account has been compromised..." | Spam | 0.96 |
| "Reminder: Your appointment is scheduled for tomorrow..." | Not-Spam | 0.00 |
| "FREE entry in a weekly competition to win an iPad..." | Spam | 1.00 |

---

## 📁 Project Structure

> Adjust this to match your actual repo layout before publishing.

```
sms-spam-classifier/
├── sms_spam_collection/
│   └── SMSSpamCollection          # Raw tab-separated dataset
├── spam_classification.ipynb      # Full notebook: EDA → preprocessing → training → evaluation
├── spam_detection_model.joblib    # Saved, ready-to-use trained pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/sms-spam-classifier.git
cd sms-spam-classifier

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`:**

```
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
joblib
```

On first run, download the required NLTK data:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

---

## 🚀 Usage

### Run the full pipeline

Open and run `spam_classification.ipynb` top to bottom — it covers data loading, cleaning, feature extraction, `GridSearchCV` tuning, evaluation plots, and saving the model to `spam_detection_model.joblib`.

### Load the saved model and classify new messages

```python
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

model = joblib.load("spam_detection_model.joblib")

new_message = "Win a free iPhone now! Click here to claim."
processed = preprocess_message(new_message)
prediction = model.predict([processed])[0]
proba = model.predict_proba([processed])[0]

print("Spam" if prediction == 1 else "Not-Spam", f"(confidence: {proba[prediction]:.2f})")
```

---

## 🔮 Future Improvements

- Swap `CountVectorizer` for **TF-IDF** to down-weight very common terms automatically.
- Try discriminative models (Logistic Regression, Linear SVM) for comparison.
- Address class imbalance explicitly (e.g. class weighting, SMOTE) rather than relying solely on F1-based tuning.
- Wrap the saved model in a small **Flask/FastAPI** service for real-time inference.
- Expand beyond bigrams or add character n-grams to catch obfuscated spam (e.g. `"fr33"`, `"w1n"`).

---

## 🙏 Acknowledgments

- **SMS Spam Collection Dataset** — a public, widely used benchmark for SMS spam research.
- **NLTK** for tokenization, stopwords, and stemming.
- **scikit-learn** for `CountVectorizer`, `MultinomialNB`, `Pipeline`, and `GridSearchCV`.

---

## 📄 License

Add a license (e.g. MIT) here if you intend for others to reuse this code.
