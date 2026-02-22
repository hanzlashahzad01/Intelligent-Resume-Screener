import os
import random
import numpy as np
import matplotlib.pyplot as plt

from nlp.preprocess import clean_text
from nlp.vectorize import get_tfidf_matrix, get_count_matrix
from models.matcher import calculate_tfidf_similarity, calculate_semantic_similarity
from sklearn.metrics.pairwise import cosine_similarity

def run_elite_evaluation():
    print("--- 🔬 ELITE Evaluation Suite ---")
    print("Seeded for reproducibility (random=42, numpy=42)\n")
    
    jd = "Expert Python Developer with Deep Learning experience"
    cvs = [
        "Senior Python Engineer specializing in Neural Networks and PyTorch", # Semantic Match
        "Python Coder who likes building simple websites with Flask",       # Keyword Match (Low context)
        "Java expert with focus on Spring Boot and Docker",                 # No Match
    ]
    candidate_labels = ["Semantic (ML)", "Keyword (Flask)", "Irrelevant (Java)"]
    
    # Fix seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Preprocess
    cleaned_cvs = [clean_text(cv) for cv in cvs]
    cleaned_jd = clean_text(jd)
    
    # 1️⃣ TF-IDF
    tfidf_matrix, _ = get_tfidf_matrix([cleaned_jd] + cleaned_cvs)
    scores_tfidf = calculate_tfidf_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # 2️⃣ CountVectorizer (Ablation study: TF-IDF vs Count)
    count_matrix, _ = get_count_matrix([cleaned_jd] + cleaned_cvs)
    scores_count = cosine_similarity(count_matrix[0:1], count_matrix[1:]).flatten()
    
    # 3️⃣ Semantic (Elite, Sentence-BERT)
    scores_semantic = calculate_semantic_similarity(jd, cvs)
    
    print(f"\n{'Candidate':<40} | {'CountVec':<9} | {'TF-IDF':<9} | {'Semantic (SBERT)':<15}")
    print("-" * 90)
    for i in range(len(cvs)):
        print(
            f"{cvs[i][:38]:<40} | "
            f"{scores_count[i]:.4f}  | "
            f"{scores_tfidf[i]:.4f}  | "
            f"{scores_semantic[i]:.4f}"
        )
    
    # --- Visual Plot Generation ---
    try:
        x = np.arange(len(candidate_labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width, scores_count, width, label='Count Vectorizer', color='#A3C1AD')
        rects2 = ax.bar(x, scores_tfidf, width, label='TF-IDF (Keyword)', color='#34495E')
        rects3 = ax.bar(x + width, scores_semantic, width, label='Semantic (SBERT)', color='#E74C3C')

        ax.set_ylabel('Similarity Score')
        ax.set_title('Ablation Study: Matching Techniques Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(candidate_labels)
        ax.legend()

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join("static", "evaluation_results.png")
        os.makedirs("static", exist_ok=True)
        plt.savefig(plot_path)
        print(f"\n✅ Visual evaluation plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate plot: {e}")

    print("\n[Observation]")
    print("- TF-IDF typically outperforms simple CountVectorizer by down-weighting generic buzzwords.")
    print("- Semantic Matching (SBERT) can capture relationships like 'Neural Networks' ≈ 'Deep Learning'")
    print("  even when exact keywords are not present in both texts.")

if __name__ == "__main__":
    try:
        run_elite_evaluation()
    except Exception as e:
        print(
            "\n⚠️ Note: SBERT / spaCy / model dependencies might still be downloading or installing. "
            "Please ensure models are available before re-running evaluation."
        )
        print(f"Error detail: {e}")
