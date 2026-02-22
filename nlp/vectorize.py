from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

def get_tfidf_matrix(texts):
    """
    Converts a list of texts into a TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def get_count_matrix(texts):
    """
    Converts a list of texts into a Count matrix (Ablation study).
    """
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(texts)
    return count_matrix, vectorizer
