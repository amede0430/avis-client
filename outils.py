import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation



def vecteurs_avis(texts):
    # Télécharger la liste de stopwords français si ce n'est pas déjà fait
    nltk.download('stopwords')

    # Charger les stopwords français
    stopwords_fr = stopwords.words('french')

    # Initialiser le CountVectorizer avec la liste de stopwords en français
    vectorizer = CountVectorizer(stop_words=stopwords_fr, max_features=5000)

    # Transformer les avis en matrice document-terme
    doc_term_matrix = vectorizer.fit_transform(texts)

    # Affichage des mots les plus fréquents
    print("Mots les plus fréquents après suppression des stopwords : ")
    print(vectorizer.get_feature_names_out()) # Afficher les mots les plus fréquents
    
    return (doc_term_matrix, vectorizer)  # Retourner la matrice document-terme et le vectorizer




def sujets_avis(matrix, vectorizer):
    # Définir le nombre de sujets à extraire (ex: 5 sujets)
    n_topics = 5

    # Initialiser LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    # Appliquer LDA sur la matrice document-terme
    lda.fit(matrix)
    
    # Récupérer les mots-clés pour chaque sujet
    terms = vectorizer.get_feature_names_out() # Récupérer les mots-clés

        
    return (lda, terms)  # Retourner les sujets et les mots-clés terms




def scores_avis(texts):
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
    
    
    
def lire_avis(filename):
    df = pd.read_csv(filename, encoding="utf-8")  # Assure-toi que le fichier est dans le même dossier que ton script
    texts = df['text']
    return texts.head(5)  # Retourner les 5 premiers avis pour vérification

    
    
    
    

