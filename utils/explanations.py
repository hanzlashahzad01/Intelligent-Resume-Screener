import re
import math
from typing import List
from nlp.preprocess import clean_text

def get_top_matching_sentences(
    jd_text: str,
    cv_text: str,
    top_k: int = 3,
    language: str = "english",
) -> List[str]:
    """
    Returns top-k CV sentences that best match the JD based on token overlap.
    Improved handling for messy/concatenated resume text.
    """
    if not jd_text or not cv_text or top_k <= 0:
        return []

    # Pre-clean the CV text for better splitting
    # Replace form feeds and other control characters with newlines
    cv_text_clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '\n', cv_text)
    
    cleaned_jd = clean_text(jd_text, language=language)
    jd_tokens = set(cleaned_jd.split())
    if not jd_tokens:
        return []

    # Better split: sentences, newlines, and bullet points
    # Split by: . ! ? following by space OR multiple newlines OR bullet points
    sentences_raw = re.split(r"(?<=[.!?])\s+|\n+|[•\*\-➢]\s+", cv_text_clean)

    scored_sentences = []
    seen_content = set()

    for sent in sentences_raw:
        # Normalize whitespace
        sent = re.sub(r'\s+', ' ', sent).strip()
        
        # Filter out very short or degenerate snippets
        if len(sent) < 15 or len(sent) > 300:
            continue
            
        # Avoid duplicate snippets (case-insensitive)
        norm_sent = sent.lower()
        if norm_sent in seen_content:
            continue
        seen_content.add(norm_sent)

        cleaned_sent = clean_text(sent, language=language)
        tokens = cleaned_sent.split()
        if not tokens:
            continue

        overlap = jd_tokens.intersection(tokens)
        if not overlap:
            continue

        # Score based on overlap ratio + raw count to favor meaningful sentences
        # Using a slight boost for actual token count to avoid very short "matches"
        score = (len(overlap) / len(tokens)) * (math.log(len(overlap) + 1))
        scored_sentences.append((score, sent))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    
    # Return unique top sentences
    return [s for _, s in scored_sentences[:top_k]]

