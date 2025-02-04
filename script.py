import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Charger le fichier CSV
df = pd.read_csv("TP_pets_reviews.csv", encoding="utf-8")  # Assure-toi que le fichier est dans le même dossier que ton script

# Récupérer les éléments de la colonne text
texts = df['text']

# Initialiser le vectorizer
vectorizer = TfidfVectorizer()

# Transformer les avis en matrice TF-IDF
tfidf_matrix = vectorizer.fit_transform(texts)

# Affichage des mots importants
print("Mots clés du vocabulaire : ")
print(vectorizer.get_feature_names_out())

# Affichage des scores TF-IDF
print("\nMatrice TF-IDF : ")
print(tfidf_matrix)



nltk.download('vader_lexicon') # Télécharger le lexique VADER
sia = SentimentIntensityAnalyzer() # Initialiser le sentiment analyzer

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"Avis : {text}")
    print(f"Scores : {scores}")  # Détails des scores
    if scores['compound'] >= 0.05:
        print("Sentiment : Positif 😃")
    elif scores['compound'] <= -0.05:
        print("Sentiment : Négatif 😡")
    else:
        print("Sentiment : Neutre 😐")
    print("-" * 50)
