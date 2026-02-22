import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data for both languages
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

def clean_text(text, language='english'):
    """
    Cleans the input text supporting English and German.
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Combined Stopwords (English + German) for multilingual support
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('german')))
    
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    return " ".join(filtered_tokens)
