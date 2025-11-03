import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Φόρτωση του dataset
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Δημιουργία λίστας από patterns και αντιστοίχιση με tags
patterns = []
tags = []
responses_by_tag = {}

for intent in data["intents"]:
    tag = intent["tag"]
    responses_by_tag[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(tag)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match = similarities.argmax()
    confidence = similarities[0][best_match]

    if confidence < 0.3:
        return "Δεν κατάλαβα. Μπορείς να το πεις αλλιώς;"
    
    tag = tags[best_match]
    return random.choice(responses_by_tag[tag])

def chat():
    print("🤖 Iosifidis Dynamics AI ChatBot ενεργοποιήθηκε! Πληκτρολόγησε 'έξοδος' για να τερματίσεις.")
    while True:
        user_input = input("Εσύ: ")
        if user_input.lower() == "έξοδος":
            print("Bot: Αντίο! 👋")
            break
        response = get_response(user_input.lower())
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
