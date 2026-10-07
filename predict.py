import pickle

# Model & Vectorizer load cheyadam
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Testing news
news_text = input("Enter news text to check: ")

# Vectorization & Prediction
news_vector = vectorizer.transform([news_text])
prediction = model.predict(news_vector)

print(f"Result: {prediction[0]}")