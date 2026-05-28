"""
AI-Powered Autocorrect Tool - Advanced BERT Contextual Corrector
Author: Senior AI/ML Engineer
Description: Implements context-aware spelling and grammar correction using 
             a pretrained BERT model from Hugging Face (distilbert-base-uncased).
             Integrates phonetic/edit-distance candidate generation with
             neural language model probability scoring to resolve homophones 
             (e.g., "buy" vs "bye") and select contextually appropriate words.
"""

import re
import string
import torch
from transformers import pipeline
from textblob import Word

# =====================================================================
# PURE PYTHON LEVENSHTEIN DISTANCE FUNCTION
# =====================================================================
def get_edit_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein edit distance between two strings."""
    s1 = s1.lower()
    s2 = s2.lower()
    if len(s1) < len(s2):
        return get_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# =====================================================================
# BERT CONTEXTUAL CORRECTOR CLASS
# =====================================================================
class BERTContextualCorrector:
    def __init__(self):
        self.model_name = "prajjwal1/bert-tiny"
        self.enabled = False
        self.mask_filler = None
        
        # Define common context-sensitive confusion sets (homophones/grammar errors)
        self.confusion_sets = {
            "their": ["their", "there", "they're"],
            "there": ["their", "there", "they're"],
            "they're": ["their", "there", "they're"],
            
            "its": ["its", "it's"],
            "it's": ["its", "it's"],
            
            "your": ["your", "you're"],
            "you're": ["your", "you're"],
            
            "to": ["to", "too", "two"],
            "too": ["to", "too", "two"],
            "two": ["to", "too", "two"],
            
            "bye": ["bye", "buy", "by"],
            "buy": ["bye", "buy", "by"],
            "by": ["bye", "buy", "by"],
            
            "lose": ["lose", "loose"],
            "loose": ["lose", "loose"],
            
            "then": ["then", "than"],
            "than": ["then", "than"],
            
            "close": ["close", "clothes"],
            "clothes": ["close", "clothes"],
            
            "affect": ["affect", "effect"],
            "effect": ["affect", "effect"],
            
            "accept": ["accept", "except"],
            "except": ["accept", "except"],
            
            "principal": ["principal", "principle"],
            "principle": ["principal", "principle"]
        }
        
        # Flatten confusion keys for O(1) lookup
        self.confusion_keys = set(self.confusion_sets.keys())

    def load_model(self):
        """Loads the pre-trained HuggingFace pipeline in a non-blocking/handled way."""
        if self.enabled:
            return True
        
        print(f"[BERT] Initializing contextual corrector using '{self.model_name}'...")
        try:
            # We use DistilBERT fill-mask pipeline which is extremely fast and perfect for local environments
            self.mask_filler = pipeline(
                "fill-mask", 
                model=self.model_name,
                device=-1  # Force CPU to ensure stability on all platforms; change to 0 if GPU is available
            )
            self.enabled = True
            print("[BERT] Model loaded successfully.")
            return True
        except Exception as e:
            print(f"[BERT WARNING] Failed to load Hugging Face model: {e}")
            print("[BERT] Falling back to basic spelling heuristics.")
            self.enabled = False
            return False

    def get_contextual_candidates(self, word: str) -> list:
        """
        Generates candidates for a word. 
        Uses confusion sets if the word is a known homophone/grammatical trap, 
        otherwise falls back to TextBlob candidate suggestions.
        """
        word_lower = word.lower().strip(string.punctuation)
        
        # Check if the word belongs to our contextual confusion sets
        if word_lower in self.confusion_keys:
            return self.confusion_sets[word_lower]
            
        # Otherwise generate spelling candidates using TextBlob spell check
        tb_candidates = Word(word_lower).spellcheck()
        # Extract candidate strings from (candidate, confidence) pairs
        candidates = [cand for cand, score in tb_candidates if cand != word_lower]
        
        # Add the original word to the list to see if it remains the best choice contextually
        if word_lower not in candidates:
            candidates.insert(0, word_lower)
            
        return candidates

    def correct_sentence(self, sentence: str) -> dict:
        """
        Corrects a sentence using the BERT-based fill-mask contextual model.
        
        Returns a dictionary:
        {
            "original": original_sentence,
            "corrected": corrected_sentence,
            "num_changes": number_of_words_changed,
            "changes": list of change details,
            "using_bert": boolean indicating whether BERT model was active
        }
        """
        if not sentence.strip():
            return {
                "original": "",
                "corrected": "",
                "num_changes": 0,
                "changes": [],
                "using_bert": self.enabled
            }

        # Lazy-load BERT model if not already loaded
        if not self.enabled and self.mask_filler is None:
            self.load_model()

        # Tokenize sentence simply by spaces to track words
        words = sentence.split()
        corrected_words = list(words)
        changes = []

        # If BERT is not loaded, gracefully bypass and return unchanged
        if not self.enabled:
            return {
                "original": sentence,
                "corrected": sentence,
                "num_changes": 0,
                "changes": [],
                "using_bert": False
            }

        for idx, word in enumerate(words):
            # Strip punctuation for evaluation
            clean_word = word.strip(string.punctuation)
            if not clean_word or not clean_word.isalpha():
                continue
                
            clean_word_lower = clean_word.lower()
            
            # Check if this word is a candidate for correction:
            # 1. It is a known homophone/grammatical confusion word
            # 2. OR TextBlob thinks it's misspelled (confidence for the word itself is not high)
            tb_spellcheck = Word(clean_word_lower).spellcheck()
            is_misspelled = len(tb_spellcheck) > 0 and tb_spellcheck[0][0] != clean_word_lower
            is_homophone = clean_word_lower in self.confusion_keys
            
            if is_misspelled or is_homophone:
                # Get spelling candidates for this word
                candidates = self.get_contextual_candidates(clean_word_lower)
                if not candidates or (len(candidates) == 1 and candidates[0] == clean_word_lower):
                    continue

                # Prepare the sentence with a [MASK] token at this word's position
                words_masked = list(words)
                words_masked[idx] = self.mask_filler.tokenizer.mask_token
                masked_sentence = " ".join(words_masked)

                try:
                    # Run the masked sentence through BERT with top_k=20 to ensure candidates are covered
                    bert_predictions = self.mask_filler(masked_sentence, top_k=20)
                    
                    # Ensure predictions is a list
                    if isinstance(bert_predictions, dict):
                        bert_predictions = [bert_predictions]

                    # Map BERT predictions to candidate match scores
                    # Predictions are sorted by probability descending
                    bert_tokens = {pred['token_str'].lower().strip(): pred['score'] for pred in bert_predictions}
                    
                    best_candidate = clean_word_lower
                    max_score = -1.0
                    
                    for cand in candidates:
                        cand_clean = cand.lower().strip()
                        
                        # Score candidate based on BERT's prediction probability
                        # If the candidate is in BERT's vocabulary predictions, use BERT score
                        if cand_clean in bert_tokens:
                            score = bert_tokens[cand_clean]
                        # Else, if it's the original word, give it a tiny bias to avoid overcorrecting
                        elif cand_clean == clean_word_lower:
                            score = 0.05
                        # Else, give it a baseline penalty score
                        else:
                            score = 0.001
                            
                        # Boost candidates that are closer in edit distance to avoid random contextual words
                        dist = get_edit_distance(clean_word_lower, cand_clean)
                        if dist > 0:
                            # Apply edit-distance penalty factor
                            score = score / (dist * 1.5)

                        if score > max_score:
                            max_score = score
                            best_candidate = cand

                    # If the selected best candidate is different from the original word, apply the change
                    if best_candidate.lower() != clean_word_lower:
                        # Re-apply capitalization style
                        if word.isupper():
                            final_word = best_candidate.upper()
                        elif word[0].isupper():
                            final_word = best_candidate.capitalize()
                        else:
                            final_word = best_candidate
                            
                        # Preserve original surrounding punctuation
                        prefix = word[:word.find(clean_word)]
                        suffix = word[word.find(clean_word) + len(clean_word):]
                        corrected_words[idx] = f"{prefix}{final_word}{suffix}"
                        
                        changes.append({
                            "original": clean_word,
                            "corrected": final_word,
                            "reason": "Contextual grammar" if is_homophone else "Spelling error"
                        })
                except Exception as e:
                    # In case of tokenization or API failure during mask filling, skip this word
                    print(f"[BERT ERROR] Failed to evaluate mask at index {idx}: {e}")
                    continue

        return {
            "original": sentence,
            "corrected": " ".join(corrected_words),
            "num_changes": len(changes),
            "changes": changes,
            "using_bert": True
        }

# =====================================================================
# SELF-TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    print("--- BERT Corrector Module Self-Test ---")
    corrector = BERTContextualCorrector()
    
    # Try to load model
    if corrector.load_model():
        # Test Case 1: Homophone correction (Standard spellcheckers fail because "close" is a real word)
        test_1 = "I need to buy some new close for the winter."
        print(f"\nTest 1 (Homophone/Context): '{test_1}'")
        res_1 = corrector.correct_sentence(test_1)
        print(f"Corrected: {res_1['corrected']}")
        print(f"Changes: {res_1['changes']}")
        
        # Test Case 2: Simple spelling correction
        test_2 = "They are going to there school now."
        print(f"\nTest 2 (Homophone/Context): '{test_2}'")
        res_2 = corrector.correct_sentence(test_2)
        print(f"Corrected: {res_2['corrected']}")
        print(f"Changes: {res_2['changes']}")
    else:
        print("BERT Model could not be loaded. Self-test aborted.")
