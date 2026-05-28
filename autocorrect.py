"""
AI-Powered Autocorrect Tool - TextBlob Basic Autocorrect Module
Author: Senior AI/ML Engineer
Description: Implements basic spelling correction using the TextBlob library. 
             Provides word correction, sentence correction, and tracking of changes.
"""

from textblob import TextBlob
from textblob import Word
import re

# =====================================================================
# TEXTBLOB CORRECTOR CLASS
# =====================================================================
class TextBlobCorrector:
    def __init__(self):
        pass

    def correct_word(self, word: str) -> str:
        """
        Corrects a single word using TextBlob.
        """
        # Clean word from surrounding punctuation before correcting
        cleaned_word = re.sub(r'[^\w]', '', word)
        if not cleaned_word:
            return word
        
        corrected = str(Word(cleaned_word).correct())
        
        # Preserve original capitalization style (Title, UPPER, lower)
        if word.isupper():
            return corrected.upper()
        elif word[0].isupper():
            return corrected.capitalize()
        return corrected

    def get_word_candidates(self, word: str) -> list:
        """
        Returns a list of spelling candidates and their confidence scores.
        Example: [('hello', 0.95), ('help', 0.05)]
        """
        cleaned_word = re.sub(r'[^\w]', '', word)
        if not cleaned_word:
            return [(word, 1.0)]
        
        # Word.spellcheck() returns a list of (candidate, confidence) pairs
        candidates = Word(cleaned_word).spellcheck()
        return candidates

    def correct_sentence(self, sentence: str) -> dict:
        """
        Corrects a complete sentence using TextBlob's built-in correct() method.
        Compares the original sentence to the corrected one to identify specific changes.
        
        Returns a dictionary:
        {
            "original": original_sentence,
            "corrected": corrected_sentence,
            "num_changes": number_of_words_changed,
            "changes": list of dictionaries with original and corrected words
        }
        """
        if not sentence.strip():
            return {
                "original": "",
                "corrected": "",
                "num_changes": 0,
                "changes": []
            }

        # TextBlob's standard sentence correction
        blob = TextBlob(sentence)
        corrected_blob = blob.correct()
        corrected_sentence = str(corrected_blob)

        # Let's perform a word-by-word alignment to extract which words were changed
        original_words = sentence.split()
        corrected_words = corrected_sentence.split()
        
        changes = []
        
        # If lengths match, compare word-by-word (most common case for spelling fixes)
        if len(original_words) == len(corrected_words):
            for orig, corr in zip(original_words, corrected_words):
                # Strip punctuation for comparison
                orig_clean = re.sub(r'[^\w]', '', orig).lower()
                corr_clean = re.sub(r'[^\w]', '', corr).lower()
                
                if orig_clean != corr_clean and orig_clean:
                    changes.append({
                        "original": re.sub(r'[^\w]', '', orig),
                        "corrected": re.sub(r'[^\w]', '', corr)
                    })
        else:
            # Fallback if sentence length changed due to splitting/merging
            # Find distinct words that differ
            for orig in original_words:
                orig_clean = re.sub(r'[^\w]', '', orig)
                if not orig_clean:
                    continue
                corr_val = self.correct_word(orig_clean)
                if orig_clean.lower() != corr_val.lower():
                    changes.append({
                        "original": orig_clean,
                        "corrected": corr_val
                    })

        return {
            "original": sentence,
            "corrected": corrected_sentence,
            "num_changes": len(changes),
            "changes": changes
        }

# =====================================================================
# SELF-TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    print("--- TextBlob Corrector Module Self-Test ---")
    corrector = TextBlobCorrector()
    
    test_word = "necesary"
    print(f"Word spelling candidates for '{test_word}':")
    print(corrector.get_word_candidates(test_word))
    print(f"Corrected word: {corrector.correct_word(test_word)}")
    
    test_sentence = "I think this is a verry bad spellling error."
    result = corrector.correct_sentence(test_sentence)
    print("\nSentence Correction Result:")
    print(f"Original:  {result['original']}")
    print(f"Corrected: {result['corrected']}")
    print(f"Changes count: {result['num_changes']}")
    print("List of changes:")
    for change in result['changes']:
        print(f"  {change['original']} -> {change['corrected']}")
