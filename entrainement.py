import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Charger le modèle pré-entraîné de CamemBERT pour la classification de texte
model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Utiliser le pipeline de Hugging Face pour la classification de texte
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Charger le fichier CSV et diviser les données en ensembles d'entraînement et de test
def charger_donnees(filename):
    # Lire le fichier CSV
    df = pd.read_csv(filename, encoding="utf-8")
    
    # Supprimer les lignes vides
    df = df.dropna()

    # Appliquer BERT sur chaque avis pour prédire le sentiment
    sentiments = []
    for text in df["text"]:
        # Utilisation du modèle pré-entraîné CamemBERT pour prédire le sentiment
        result = classifier(text)[0]  # Analyse de sentiment avec BERT
        label = result['label']  # 'LABEL_0', 'LABEL_1', ou 'LABEL_2'

        # Convertir en 0 = négatif, 1 = neutre, 2 = positif
        if label == 'LABEL_0':  # Négatif
            sentiments.append(0)
        elif label == 'LABEL_1':  # Neutre
            sentiments.append(1)
        else:  # Positif
            sentiments.append(2)

    df["label"] = sentiments  # Ajouter la colonne des labels

    # Diviser les données (80% entraînement, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df

def evaluer_modele(model, tokenizer, test_df):
    inputs = tokenizer(list(test_df["text"]), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(test_df["label"].values)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

def visualiser_evaluation(model, tokenizer, test_df):
    # Tokenisation
    inputs = tokenizer(list(test_df["text"]), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(test_df["label"].values)

    # Prédictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Matrice de confusion
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Négatif", "Neutre", "Positif"], yticklabels=["Négatif", "Neutre", "Positif"])
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.title("Matrice de Confusion")
    plt.show()

    # Affichage des métriques sous forme de barres
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
    
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=["blue", "green", "orange", "red"])
    plt.ylim(0, 1)
    plt.title("Scores du Modèle")
    plt.show()
