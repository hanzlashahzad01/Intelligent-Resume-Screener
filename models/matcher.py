try:
    from sentence_transformers import SentenceTransformer, util
    # Load a lightweight, fast model suitable for local CPU
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
except Exception as e:
    print(f"⚠️ Warning: Semantic Search (SBERT) could not be initialized: {e}")
    model = None
    util = None

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_tfidf_similarity(jd_vector, cv_vectors):
    similarities = cosine_similarity(jd_vector, cv_vectors)
    return similarities.flatten()

def calculate_semantic_similarity(jd_text, cv_texts):
    """
    Calculates semantic similarity using Sentence-BERT.
    Provides context-aware matching.
    """
    if model is None:
        return np.zeros(len(cv_texts))
    
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    cv_embeddings = model.encode(cv_texts, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(jd_embedding, cv_embeddings)
    return cosine_scores.cpu().numpy().flatten()
