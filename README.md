# AI-Powered Autocorrect Tool

An advanced, context-aware Spelling and Grammar Correction System that merges traditional heuristic-based Natural Language Processing with deep learning transformers. This project showcases the difference between word-level dictionary lookups (TextBlob) and neural-network-based contextual understanding (DistilBERT).

Developed as a highly professional, submission-ready project for internships, training programs, and academic presentations.

---

## 🌟 Features

1. **Basic Spell Correction (TextBlob)**:
   - Word-level spell check based on frequency dictionaries and Levenshtein edit distance.
   - Fast, lightweight correction of obvious typographical slips.
   
2. **NLP Preprocessing & Cleaning (NLTK)**:
   - Complete sentence and word tokenization using `nltk.tokenize`.
   - Text cleaning (spaces and special character normalization).
   - Analysis of text structure (character counts, word counts, stopword density, and reading analytics).

3. **Advanced AI Contextual Correction (BERT)**:
   - Utilizes HuggingFace `transformers` and a pre-trained **DistilBERT** language model.
   - Employs a custom-designed **Fill-Mask Pipeline** to evaluate word probability in context.
   - Resolves tricky homophones and grammatical traps (e.g., *their/there/they're*, *close/clothes*, *lose/loose*, *bye/buy/by*) which standard dictionary checkers completely miss.

4. **Premium UI Dashboard**:
   - **Glassmorphism Design**: High-fidelity dark mode interface featuring frosted cards, backdrop blurs, glow borders, and floating background blobs.
   - **Interactive Tests**: Preloaded sample test cases that can be loaded in one click.
   - **Visual Diffs**: Side-by-side comparison tables highlighting exactly what was changed (red wavy lines for original errors, green tags for corrections).
   - **Real-Time Analytics**: Visual indicators of processing speeds, BERT correction counts, and an input spelling quality score.
   - **Lazy-Loading**: The application opens instantaneously; the deep-learning weights load asynchronously on demand.

5. **Exhaustive Documentation & Materials**:
   - Academic mini-project report (`report_content.md`).
   - Presentation PPT slides structure (`ppt_content.md`).
   - Batch of 20 ready-to-test sentences (`sample_inputs.txt`).

---

## 🛠️ Architecture & Workflow

Traditional spelling checkers look at each word in isolation. For example, in the sentence *"I want to buy some close"*, the word **close** is spelled correctly, so a dictionary checker leaves it untouched. 

Our hybrid model solves this by running a **two-pass evaluation**:

```
[User Input Text]
       │
       ▼
[NLTK Preprocessor] ──(Tokenization & Clean)
       │
       ├──► [TextBlob Corrector] ──► Generate Spelling Candidates (Edit Distance)
       │                                       │
       │                                       ▼
       └─────────────────────────────► [BERT Context Evaluator]
                                               │
                                      (Is word a homophone or
                                      misspelled in context?)
                                               │
                                               ▼
                                      Mask Word: "[MASK]"
                                               │
                                               ▼
                                      Evaluate probabilities
                                      via DistilBERT fill-mask
                                               │
                                               ▼
                                   [Select Highest Scoring
                                     Intersecting Candidate]
                                               │
                                               ▼
                                   [Dashboard Output & Metrics]
```

---

## 💻 Technologies & Stack

- **Language**: Python 3.11+
- **Deep Learning Framework**: PyTorch (`torch`)
- **Transformer Engine**: HuggingFace `transformers` (DistilBERT)
- **Natural Language Toolkits**: NLTK, TextBlob
- **Web App Server**: Flask (HTML5, Custom CSS3 Grid, JavaScript ES6)
- **Data Containers**: Pandas, NumPy

---

## 🚀 Installation & Execution

### 1. Prerequisites
Ensure you have **Python 3.11** or higher installed. You can check your version using:
```bash
python --version
```

### 2. Clone/Extract the Project
Navigate to the project root directory:
```bash
cd Autocorrect_AI_Project
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```
*Note: PyTorch and HuggingFace models may take a few minutes to download depending on your network speed.*

### 4. Run the Application
Start the Flask web server:
```bash
python app.py
```

Upon starting, you will see a console output indicating that the server is active:
```text
============================================================
      AI-POWERED AUTOCORRECT TOOL SERVER IS STARTING
============================================================
 * Preprocessor: NLTK Tokenizer initialized.
 * TextBlob: Spell checker ready.
 * BERT Model: On Standby (lazy-loads on first HTTP request).
 * Running Locally: Open http://127.0.0.1:5000 in your browser.
============================================================
```

### 5. Access the Web UI
Open your favorite browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 📝 10 Example Corrected Outputs Demonstration

Here is a curated comparison of how the tool handles spelling vs. contextual errors.

| # | Input Sentence (Errors) | TextBlob (Traditional) | BERT (Advanced Context-Aware) | Error Type Resolved |
|---|-------------------------|------------------------|-------------------------------|---------------------|
| 1 | I went to buy some **close**. | I went to buy some **close**. | I went to buy some **clothes**. | Homophone Context |
| 2 | They are going to **there** school. | They are going to **there** school. | They are going to **their** school. | Grammar Context |
| 3 | I am going to **bye** a laptop. | I am going to **bye** a laptop. | I am going to **buy** a laptop. | Homophone Context |
| 4 | It is easy to **loose** your keys. | It is easy to **loose** your keys. | It is easy to **lose** your keys. | Vocabulary Context |
| 5 | The medicine had a strong **affect**. | The medicine had a strong **affect**. | The medicine had a strong **effect**. | Confused Words |
| 6 | I think this is a **verry** bad **spellling** error. | I think this is a **very** bad **spelling** error. | I think this is a **very** bad **spelling** error. | Obvious Typo / Orthography |
| 7 | The dog wagged **its** tail because **it's** happy. | The dog wagged **its** tail because **it's** happy. | The dog wagged **its** tail because **it's** happy. | Grammar Verification |
| 8 | We **accept** your invitation **except** for the last day. | We **accept** your invitation **except** for the last day. | We **accept** your invitation **except** for the last day. | Homophone Resolution |
| 9 | The **principal** is a man of high **principles**. | The **principal** is a man of high **principles**. | The **principal** is a man of high **principles**. | Confused Nouns |
| 10| This NLP **proproject** is **simpley outtstanding**. | This NLP **project** is **simply outstanding**. | This NLP **project** is **simply outstanding**. | Severe Typos |

---

## 🔮 Future Scope

While the current implementation represents a highly effective, submission-ready project, the following modules can be added for future expansions:
1. **Sequence-to-Sequence Grammar Correction**: Incorporating model pipelines like T5 (`T5-base` fine-tuned on grammar correction datasets) to correct full sentence structures (e.g. subject-verb agreements and tense rephrasing).
2. **Multi-lingual Support**: Expanding the preprocessing pipelines to handle Spanish, French, and German using multi-lingual BERT models (`bert-base-multilingual-cased`).
3. **Desktop & Browser Extension Integrations**: Packaging the backend into a lightweight REST API that can be queried by a Chrome Extension to provide real-time autocorrect on text inputs across the web.
4. **Auto-Learning User Dictionary**: Allowing users to mark word suggestions as "ignored" or "add to custom dictionary" via the UI, saving preferences locally in a SQLite database.

---

## 📁 Screenshots Description
When presenting this project for evaluation, capture and include the following screenshots:
1. **Initial Dashboard State**: Showing the frosted glassmorphism cards, clear inputs, standby status of the BERT badge, and the welcome message.
2. **Test Case Loading**: Demonstrating the dropdown interaction and loading a sample sentence (like the "clothes" homophone).
3. **Asynchronous Loading Animation**: A screenshot of the loading overlay with the neon spinner and the status message *"Loading BERT model weights into memory..."*.
4. **Result Dashboard**: Highlighting the performance metrics cards (e.g., speed `0.235s`, `1` BERT correction, spelling quality `90%`), the word comparisons with dynamic red/green highlights, and the NLTK statistics grid.
