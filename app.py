"""
AI-Powered Autocorrect Tool - Main Flask Web Application
Author: Senior AI/ML Engineer
Description: Integrates all backend elements (preprocessing, basic autocorrect, 
             and advanced BERT corrector) into a unified web dashboard. 
             Optimized with lazy-loading model mechanics and beautiful 
             HTML-differ highlighting.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import time
import string
import os

# Import custom processing modules
from preprocessing import TextPreprocessor
from autocorrect import TextBlobCorrector
from bert_corrector import BERTContextualCorrector

# Initialize Flask application
# Static folder is for CSS, templates folder is for index.html
app = Flask(__name__, static_folder='static', template_folder='templates')

# Instantiate preprocessors and models
preprocessor = TextPreprocessor()
textblob_corrector = TextBlobCorrector()
bert_corrector = BERTContextualCorrector()

# =====================================================================
# DYNAMIC DIFFER HIGHLIGHTER
# =====================================================================
def get_highlighted_diffs(orig_text: str, corr_text: str, is_bert: bool = False) -> tuple:
    """
    Compares original text with corrected text word-by-word.
    Wraps modifications in customized HTML span tags with detailed hover titles.
    
    Returns: (highlighted_original_html, highlighted_corrected_html)
    """
    orig_tokens = orig_text.split()
    corr_tokens = corr_text.split()
    
    orig_html = []
    corr_html = []
    
    # If the token counts match, perform a precise 1-to-1 word alignment
    if len(orig_tokens) == len(corr_tokens):
        for o, c in zip(orig_tokens, corr_tokens):
            # Clean punctuation to compare raw word spelling
            o_clean = o.strip(string.punctuation)
            c_clean = c.strip(string.punctuation)
            
            if o_clean.lower() != c_clean.lower() and o_clean:
                # Different word: Highlight the original as error, the corrected as fixed
                hl_class = "highlight-bert-fixed" if is_bert else "highlight-corrected"
                orig_html.append(f'<span class="highlight-error" title="AI Suggests: {c_clean}">{o}</span>')
                corr_html.append(f'<span class="{hl_class}" title="Corrected from: {o_clean}">{c}</span>')
            else:
                orig_html.append(o)
                corr_html.append(c)
    else:
        # Fallback alignment if word lengths changed (splitting/merging words)
        for o in orig_tokens:
            o_clean = o.strip(string.punctuation).lower()
            if o_clean and o_clean not in corr_text.lower():
                orig_html.append(f'<span class="highlight-error" title="Word modified">{o}</span>')
            else:
                orig_html.append(o)
                
        for c in corr_tokens:
            c_clean = c.strip(string.punctuation).lower()
            if c_clean and c_clean not in orig_text.lower():
                hl_class = "highlight-bert-fixed" if is_bert else "highlight-corrected"
                corr_html.append(f'<span class="{hl_class}" title="New contextual insertion">{c}</span>')
            else:
                corr_html.append(c)
                
    return " ".join(orig_html), " ".join(corr_html)

# =====================================================================
# FLASK WEB ENDPOINTS
# =====================================================================

@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    """Returns the initialization status of the deep learning BERT model."""
    return jsonify({
        "bert_loaded": bert_corrector.enabled
    })

@app.route('/correct', methods=['POST'])
def correct_text():
    """
    Main API endpoint for spelling/grammar correction.
    Receives JSON body: {"text": "sentence to correct"}
    """
    start_time = time.time()
    
    # 1. Input Extraction and Validation
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "Empty text input provided"}), 400
        
    try:
        # 2. NLP Preprocessing and NLTK Stats
        cleaned_text = preprocessor.clean_text(text)
        stats = preprocessor.get_text_statistics(cleaned_text)
        
        # 3. Basic Autocorrect (TextBlob)
        tb_result = textblob_corrector.correct_sentence(cleaned_text)
        
        # 4. Context-Aware Advanced Autocorrect (BERT)
        # Lazily loads the BERT model weights if it's the first execution
        if not bert_corrector.enabled:
            bert_corrector.load_model()
            
        bert_result = bert_corrector.correct_sentence(cleaned_text)
        
        # 5. Calculate Processing Time
        elapsed_time = time.time() - start_time
        
        # 6. Generate Side-by-Side Highlighted Diffs
        # Highlight original vs. BERT (our ultimate context corrector)
        hl_orig, hl_bert = get_highlighted_diffs(cleaned_text, bert_result["corrected"], is_bert=True)
        # Highlight original vs. TextBlob
        _, hl_tb = get_highlighted_diffs(cleaned_text, tb_result["corrected"], is_bert=False)
        
        # 7. Compute Input Spelling Quality (Accuracy Estimation)
        # Formula: Percentage of words that didn't need correction by our advanced engine
        total_words = stats["word_count"]
        corrections_made = bert_result["num_changes"]
        if total_words > 0:
            spelling_quality = max(0, int((1 - (corrections_made / total_words)) * 100))
        else:
            spelling_quality = 100
            
        # 8. Pack and Return JSON Response
        response_payload = {
            "cleaned_text": cleaned_text,
            "highlighted": {
                "original": hl_orig,
                "textblob": hl_tb,
                "bert": hl_bert
            },
            "statistics": stats,
            "metrics": {
                "processing_time_sec": elapsed_time,
                "bert_corrections": corrections_made,
                "textblob_corrections": tb_result["num_changes"],
                "spelling_quality_score": spelling_quality
            },
            "bert_used": bert_corrector.enabled
        }
        
        return jsonify(response_payload)
        
    except Exception as e:
        import traceback
        print(f"[SERVER ERROR] Exception during text correction: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# =====================================================================
# STARTING THE SERVER
# =====================================================================
if __name__ == '__main__':
    # Ensure static and templates folders exist
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*60)
    print("      AI-POWERED AUTOCORRECT TOOL SERVER IS STARTING")
    print("="*60)
    print(" * Preprocessor: NLTK Tokenizer initialized.")
    print(" * TextBlob: Spell checker ready.")
    print(" * BERT Model: On Standby (lazy-loads on first HTTP request).")
    print(" * Running Locally: Open http://127.0.0.1:5000 in your browser.")
    print("="*60 + "\n")
    
    # Run the server on port 5000 (debug=True enables hot reloading)
    app.run(host='127.0.0.1', port=5000, debug=True)
