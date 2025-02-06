from outils import *

# Lire les avis
print("="*10, "Lire les avis", "="*10)
texts = lire_avis("TP_pets_reviews.csv")

# Vectoriser les avis
print("="*10, "Vectoriser les avis", "="*10)
(matrix, vectorizer) = vecteurs_avis(texts)

# Extraire les sujets
print("="*10, "Extraire les sujets", "="*10)
(lda, terms) = sujets_avis(matrix, vectorizer)

# Affichage des sujets
print("="*10, "Affichage des sujets", "="*10)
for idx, topic in enumerate(lda.components_):
    print(f"\n🔹 Sujet {idx+1}:")
    print([terms[i] for i in topic.argsort()[-10:]])  # Afficher les 10 mots les plus importants par sujet

# Analyser le sentiment des avis
print("="*10, "Analyser le sentiment des avis", "="*10)
scores_avis(texts)





