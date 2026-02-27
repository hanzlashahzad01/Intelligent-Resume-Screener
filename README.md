## Intelligent Resume Screener 🚀

An AI-powered tool to automate candidate shortlisting by matching CVs with Job Descriptions using NLP and Cosine Similarity.

Developed for scholarship portfolios (Stipendium Hungaricum, DAAD) to demonstrate applied AI and software engineering proficiency.

### 🎯 Goal (Problem)
CV + Job Description se **skill‑match score** nikaalna aur recruiter ko **top candidates shortlist** karke dikhana.

---

### 🧩 Functional Overview (What it does)
- **FR-1 Upload CV**: Multiple CVs upload (batch mode) as **PDF / TXT / DOCX**.
- **FR-2 Job Description Input**: Web UI par text box me JD paste.
- **FR-3 Text Processing**:
  - CV & JD ka **text extract** (PDF/TXT/DOCX).
  - **Cleaning**: lowercase, stopwords removal, tokenization.
  - **Vectorization**: TF‑IDF representation.
- **FR-4 Matching Engine**:
  - Cosine similarity score (0–100%) per CV.
  - Smart score = 40% Semantic + 30% TF‑IDF + 30% Skill‑Match ratio.
  - Top‑N / Top‑5 candidates ranking.
- **FR-5 Skill Extraction**:
  - Predefined skills list (Python, Java, ML, SQL, Git, Docker, etc.).
  - **Fuzzy Matching**: Matches variations like "NodeJS" vs "Node.js" for higher accuracy.
  - CV se skills highlight + **count**.
- **FR-6 Results Dashboard**:
  - Table: **Name | Match % | Skills Found | Experience | Education**.
  - Per‑candidate radar chart (Semantic / Keyword / Skills / Experience / Education).
  - Downloadable CSV report.
- **FR-7 Demo UI**:
  - Flask based web app with modern glassmorphism UI.
  - Upload page + detailed results dashboard.

---

### 🧠 Non-Functional Requirements
- **⏱️ Performance**: Designed to return results in **< 2 seconds for ~10 CVs** on a normal laptop (MiniLM SBERT model, efficient TF‑IDF pipeline).
- **🔐 Privacy**:
  - CVs are written to a temp folder only for parsing and are **deleted immediately** after processing.
  - No biometric or demographic attributes are stored or logged.
- **🧪 Reproducibility**:
  - Global seeds are fixed (`random.seed(42)`, `numpy.random.seed(42)`) in both `app.py` and `evaluate.py`.
  - Matching and evaluation are deterministic given the same inputs and environment.
- **📦 Local Setup**:
  - Runs locally with:
    ```bash
    pip install -r requirements.txt
    python app.py
    ```

---

## ScreenShots

## Register Page

<img width="1915" height="1029" alt="register" src="https://github.com/user-attachments/assets/b8189ca3-3974-486e-a274-2ed009ccab04" />

## Sign-Up-Page

<img width="1913" height="1030" alt="sign up" src="https://github.com/user-attachments/assets/6de38f9f-484e-44be-b1ea-d0ddd93e656b" />

## DashBoard

<img width="1913" height="1030" alt="d" src="https://github.com/user-attachments/assets/5db25f32-1dbc-4b50-a89c-2988f0c79c5e" />

<img width="1912" height="1032" alt="d1" src="https://github.com/user-attachments/assets/c42e4a90-f144-4c52-8f2e-10e27366335a" />

<img width="1912" height="999" alt="h" src="https://github.com/user-attachments/assets/8c144a8f-bfd0-4213-9484-6f03238d3f4a" />

<img width="1912" height="1033" alt="dwww" src="https://github.com/user-attachments/assets/39ffae1f-2ec8-4e64-8609-53a26edc058d" />



### 🏗️ Tech Stack (Exact)
- **Language**: Python 3.10+
- **Backend**: Flask
- **ML/NLP**: scikit-learn (TF‑IDF, CountVectorizer, Cosine Similarity), NLTK (tokenization + stopwords), pdfminer.six (PDF text extraction)
- **Optional / Bonus**: spaCy (NER), Sentence-Transformers SBERT (semantic matching)
- **Data**: CSV + plain text sample CVs and JDs
- **UI**: HTML + CSS (glassmorphism) + Chart.js via CDN

---

### 📂 Dataset (Synthetic / Sample Data)
- `data/sample_cvs/` – Synthetic / anonymised CV text files (software engineer, data scientist, etc.).
- `data/job_descriptions/` – Example job descriptions (e.g., Python developer).
- `data/sample_cv_data.csv` – Optional structured view of sample CVs for experimentation.

Ye sari files **demo aur scholarship portfolio** ke liye curated hain (no real personal data).

---

### 🧮 Method (TF-IDF + Cosine + Semantic)
1. **Text Extraction**
   - PDF → `pdfminer.six`
   - DOCX → `python-docx`
   - TXT → direct UTF‑8 read
2. **Preprocessing (`nlp/preprocess.py`)**
   - Lowercasing, URL & special character removal
   - Tokenization (NLTK)
   - English + German stopwords removal → multilingual support
3. **Vectorization (`nlp/vectorize.py`)**
   - TF‑IDF matrix for `[JD] + [CVs...]`
   - Optional CountVectorizer matrix for ablation study
4. **Matching Engine (`models/matcher.py`)**
   - **TF‑IDF cosine similarity** between JD and each CV.
   - **Semantic similarity** using SBERT (`paraphrase-MiniLM-L3-v2`) – gracefully falls back to zeros if model is unavailable.
5. **Skill Extraction (`utils/skills.py`)**
   - Regex-based match against predefined `SKILLS_DB`.
   - Used both for JD skills and CV skills.
6. **Smart Ranking (`app.py`)**
   - Skill gap analysis: matching vs missing skills.
   - Smart score:
     $$\text{SmartScore} = 0.4 \cdot \text{Semantic} + 0.3 \cdot \text{TFIDF} + 0.3 \cdot \text{SkillMatchRatio}$$
   - Scores scaled to **0–100%**, sorted descending.

---

### 📊 Results (Dashboard + CSV)
- **Web Dashboard** (`results.html`):
  - Summary table: **Name | Match % | Skills Found | Experience | Education**.
  - Per‑candidate profile: email, experience, AI‑generated summary, skills and missing skills.
  - Radar chart: Semantic vs Keyword vs Skills vs Experience vs Education.
- **CSV Export**:
  - `/download` route returns `elite_shortlist.csv` with:
    - `rank, name, score, experience, email, summary`

---

### 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/resume-screener.git
   cd resume-screener
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:5000`.

---

### 📊 Evaluation Metrics
Ye project accuracy ko measure karne ke liye niche metrics use karta hai:
- **Cosine Similarity Score**:
  - Main metric for JD–CV matching (TF‑IDF based).
- **Ablation Study – TF-IDF vs CountVectorizer**:
  - `evaluate.py` me TF‑IDF vs CountVectorizer similarities compare kiye gaye hain.
  - Observation: TF‑IDF generic buzzwords ko down‑weight karke **~15% better** separation deta hai.
- **Semantic Ablation – TF-IDF vs SBERT**:
  - Same script me TF‑IDF scores ko **Sentence‑BERT semantic scores** ke saath compare kiya gaya hai.
  - Semantic model cases jaise "Neural Networks" ≈ "Deep Learning" ko catch karta hai.
- **Precision@5 (manual)**:
  - Internal labelled test set par **Precision@5 ≈ 0.80+** (Top‑5 recommendations mostly JD ke relevant candidates hote hain).
- **Skill Extraction Accuracy**:
  - 70+ industry‑standard technical skills list ke against manual verification.

#### 🚀 How to Run Evaluation & Testing
Console me evaluation / ablation outputs aur visual plots dekhne ke liye:
```bash
python evaluate.py
```
Unit tests run karne ke liye (Engineering Excellence):
```bash
pytest tests/
```

---

### 🧪 Test Cases (Verified)
- **TC-1:** JD = `"Python ML SQL"` → Python / ML / SQL wale CVs consistently top ranks par aate hain.
- **TC-2:** Empty JD input → graceful error notification (flash message) and no processing.
- **TC-3:** Corrupt ya non‑text PDF → text extraction fail hone par user ko error message milta hai (`"Could not extract text from the provided files."`).
- **TC-4:** Duplicate CV uploads → cleaned text hash ke basis par **duplicate CVs detect ho kar skip** ho jaate hain (no double counting).

---

### 📂 Repo Structure
```
resume-screener/
│
├─ app.py                 # Flask app & routing
├─ requirements.txt       # Dependencies
├─ README.md              # Documentation
│
├─ data/                  # Sample data storage
├─ nlp/                   # Cleaning & Vectorization logic
├─ models/                # Matching engine
├─ utils/                 # Skills list & Extractor utils
├─ templates/             # HTML views
└─ static/                # CSS & Visual assets
```

---

### ⚖️ Ethics & Privacy (Europe‑Friendly)
- **No biometric data**: Koi images, face data, ya biometric features process nahi hote.
- **Skills‑based matching**:
  - Matching logic strictly **skills, experience text, education** par based hai.
  - Demographic attributes (gender, age, nationality, etc.) ignore kiye jaate hain.
- **Temporary storage only**:
  - CV files OS ke temp folder me short time ke liye store hote hain.
  - Parsing ke turant baad files **delete** ho jaati hain.
- **Bias & Ethics Engine**:
  - `utils/ethics.detect_bias` JD/CV ke andar gendered / ageist / exclusionary language detect karta hai
    (e.g., "rockstar", "ninja", "native speaker", "recent graduate").
  - **Neutral Suggestions**: System biased words ke liye professional alternatives suggest karta hai.
- **Future work (fairness)**:
  - Explicit bias‑mitigation metrics (e.g., group fairness scores).
  - Audit logs for explainability.

---

### 🚧 Limitations
- Semantic matching quality **Sentence‑Transformers model** par depend karta hai; offline / low‑resource environments me semantic part zero vector pe fallback ho sakta hai.
- Sample dataset limited size ka hai, isliye **Precision@5** aur other metrics mainly internal / illustrative hain.
- Highly non‑English / domain‑specific CVs (e.g., legal, medical) ke liye skills list aur preprocessing ko extend karna padega.

---

### 🎁 Future Work (LLMs, Multilingual, Semantic Search)
- **LLM‑based ranking**:
  - GPT / similar large language models se richer candidate summaries aur justification.
- **Multilingual CVs**:
  - Already EN + DE stopwords support, future: more languages + multilingual sentence embeddings.
- **Advanced Ablation / Research Angle**:
  - TF‑IDF vs dense embeddings vs hybrid models.
  - Bias mitigation strategies and fairness metrics integration.

---

### 🌍 Impact on Europe Scholarship Applications
This project is strategically designed to impress reviewers for scholarships like **Stipendium Hungaricum** and **DAAD**:
- **Applied AI:** Demonstrates the ability to transform theoretical NLP (TF-IDF, Cosine Similarity) into a functional tool.
- **Software Engineering:** Showcases clean architecture, modular design, and full-stack integration (Flask + ML).
- **Ethics Conscious:** Addresses privacy and bias, which are critical themes in European tech research and funding.
- **Problem Solving:** Directly addresses a real-world business need (Efficient Recruitment).
  
## 📬 Connect with Me
I'm always open to discussing new projects, creative ideas, or opportunities to be part of your visions.

Email: hanzlashahzadhanzlashahzad@gmail.com
---
*Created with ❤️ for international scholarship applications.*
