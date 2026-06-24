import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Bigger training dataset
data = [
("excellent product","Positive"),
("amazing quality","Positive"),
("very happy","Positive"),
("worth buying","Positive"),
("super experience","Positive"),
("loved it","Positive"),
("fantastic service","Positive"),
("good quality","Positive"),
("best purchase","Positive"),
("awesome item","Positive"),
("great product","Positive"),
("highly recommend","Positive"),
("perfect product","Positive"),
("nice experience","Positive"),
("excellent support","Positive"),

("worst product","Negative"),
("waste of money","Negative"),
("very bad","Negative"),
("poor quality","Negative"),
("not worth buying","Negative"),
("terrible experience","Negative"),
("hate this","Negative"),
("awful service","Negative"),
("disappointed","Negative"),
("bad purchase","Negative"),
("not useful","Negative"),
("very poor","Negative"),
("worst experience","Negative"),
("totally useless","Negative"),
("never buy again","Negative")
]

df = pd.DataFrame(data, columns=["review","sentiment"])


def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]','',text)
    return text


df["review"] = df["review"].apply(clean)

# Better vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Better model
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\nAccuracy:", round(acc*100,2), "%")

while True:

    review = input("\nEnter review: ")

    review = clean(review)

    sample = vectorizer.transform([review])

    result = model.predict(sample)

    print("Sentiment:", result[0])

    again = input("Continue? yes/no: ")

    if again.lower() != "yes":
        break