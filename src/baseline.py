from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Load IMDB dataset
ds = load_dataset("imdb")
train = pd.DataFrame(ds["train"])
test = pd.DataFrame(ds["test"])

# TF-IDF
vectorizer = TfidfVectorizer(max_features=20000)
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

y_train = train["label"]
y_test = test["label"]

# Baseline model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")

# Save report
with open("../baseline_report.md", "w") as f:
    f.write(f"# Baseline Report\n\nAccuracy: {acc}\nF1: {f1}\n")

print("Baseline complete. See baseline_report.md")
