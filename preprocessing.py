"""
AI-Powered Autocorrect Tool - NLP Preprocessing Module
Author: Senior AI/ML Engineer
Description: This script handles text cleaning, tokenization, and NLTK-based 
             preprocessing. It checks and downloads required NLTK resources 
             automatically on startup.
"""

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# =====================================================================
# NLTK RESOURCE INITIALIZATION
# =====================================================================
# We automatically download required NLTK packages if they aren't available.
# This prevents runtime errors for users when they run the project for the first time.
def initialize_nltk():
    """Verify and download required NLTK datasets."""
    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords'
    }
    
    for resource_name, resource_path in resources.items():
        try:
            # Try to load the resource to see if it's already downloaded
            nltk.data.find(resource_path)
            print(f"[NLTK] Resource '{resource_name}' already exists.")
        except LookupError:
            print(f"[NLTK] Downloading missing resource: '{resource_name}'...")
            nltk.download(resource_name, quiet=True)
            print(f"[NLTK] Resource '{resource_name}' downloaded successfully.")

# Run initialization on import so resources are ready immediately
initialize_nltk()

# =====================================================================
# TEXT PREPROCESSING CLASS
# =====================================================================
class TextPreprocessor:
    def __init__(self):
        # Retrieve the set of English stopwords for filtering
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by:
        1. Stripping leading/trailing whitespaces.
        2. Normalizing multiple consecutive spaces/newlines.
        3. Ensuring standard punctuation spacing.
        """
        if not text:
            return ""
        
        # Strip outer whitespaces
        cleaned = text.strip()
        
        # Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned

    def get_sentences(self, text: str) -> list:
        """
        Splits a block of text into sentences using NLTK's sent_tokenize.
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        return sent_tokenize(cleaned_text)

    def get_words(self, text: str) -> list:
        """
        Splits text into individual tokens (words and punctuation) using NLTK's word_tokenize.
        """
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        return word_tokenize(cleaned_text)

    def get_clean_words(self, text: str) -> list:
        """
        Extracts only alphanumeric words (removing punctuation and numbers) and converts them to lowercase.
        """
        words = self.get_words(text)
        # Filter for alphanumeric strings (words)
        return [w.lower() for w in words if w.isalnum()]

    def remove_stopwords(self, words_list: list) -> list:
        """
        Removes stopwords from a list of words.
        Useful for analytical purposes, though we keep stopwords in the full context of BERT sentences.
        """
        return [w for w in words_list if w.lower() not in self.stop_words]

    def get_text_statistics(self, text: str) -> dict:
        """
        Computes readability metrics and metadata about the input text.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "stopword_count": 0,
                "stopword_percentage": 0.0
            }
        
        sentences = self.get_sentences(cleaned)
        all_words = self.get_clean_words(cleaned)
        words_without_stops = self.remove_stopwords(all_words)
        
        stopword_count = len(all_words) - len(words_without_stops)
        stopword_percentage = (stopword_count / len(all_words) * 100) if len(all_words) > 0 else 0.0
        
        return {
            "char_count": len(cleaned),
            "word_count": len(all_words),
            "sentence_count": len(sentences),
            "stopword_count": stopword_count,
            "stopword_percentage": round(stopword_percentage, 1)
        }

# =====================================================================
# SELF-TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    print("--- Preprocessing Module Self-Test ---")
    processor = TextPreprocessor()
    test_sentence = "This is a simple test sentense! NLTK is verry powerful for tokenization."
    
    print(f"Original Text: {test_sentence}")
    print(f"Cleaned Text:  {processor.clean_text(test_sentence)}")
    print(f"Sentences:     {processor.get_sentences(test_sentence)}")
    print(f"Words:         {processor.get_words(test_sentence)}")
    print(f"Cleaned Words: {processor.get_clean_words(test_sentence)}")
    
    stats = processor.get_text_statistics(test_sentence)
    print("\nText Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
