import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\spamham.csv").fillna("")
df["Category"] = df["Category"].map({"ham": 1, "spam": 0})
x_train, x_test, y_train, y_test = train_test_split(
    df["Message"], df["Category"], test_size=.2, random_state=42)

v = TfidfVectorizer(stop_words="english")
X_train = v.fit_transform(x_train)
X_test = v.transform(x_test)
model = MultinomialNB().fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred) * 100:.4f}")
print(f"F1: {f1_score(y_test, pred, average='macro') * 100:.4f}")
print(confusion_matrix(y_test, pred))
