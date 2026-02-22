import os
import csv
import uuid
import tempfile
import random
import time
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from models.db_models import db, User, Scan
from nlp.preprocess import clean_text
from nlp.vectorize import get_tfidf_matrix
from models.matcher import calculate_tfidf_similarity, calculate_semantic_similarity
from utils.skills import extract_skills
from utils.extractor import get_text_from_file, extract_contact_info
from utils.parser import extract_entities, estimate_experience
from utils.ethics import detect_bias, generate_ai_summary
from utils.config_loader import load_config
from utils.explanations import get_top_matching_sentences

app = Flask(__name__)
app.secret_key = "elite_secret_key"
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Config & Reproducibility ---
CONFIG = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}


def calculate_bm25_scores(cleaned_jd: str, cleaned_docs: list, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """
    Lightweight BM25 implementation for a single query (JD) against a list of documents (CVs).
    """
    if not cleaned_docs:
        return np.zeros(0)

    query_tokens = cleaned_jd.split()
    docs_tokens = [doc.split() for doc in cleaned_docs]

    N = len(docs_tokens)
    doc_lengths = [len(doc) for doc in docs_tokens]
    avgdl = sum(doc_lengths) / float(N) if N > 0 else 0.0

    # Document frequency per term
    df = Counter()
    for doc in docs_tokens:
        for term in set(doc):
            df[term] += 1

    # IDF per term
    idf = {}
    for term, freq in df.items():
        # Classic BM25 IDF
        idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    scores = []
    for idx, doc in enumerate(docs_tokens):
        score = 0.0
        if not doc or doc_lengths[idx] == 0:
            scores.append(0.0)
            continue

        term_freqs = Counter(doc)
        dl = doc_lengths[idx]

        for term in query_tokens:
            if term not in term_freqs or term not in idf:
                continue
            f = term_freqs[term]
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * ((f * (k1 + 1)) / denom)

        scores.append(score)

    return np.array(scores, dtype=float)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash("Invalid username or password.")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash("Username already exists.")
            return redirect(url_for('signup'))
        
        new_user = User(username=username, email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash("Account created! Please login.")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/scan/<int:scan_id>')
@login_required
def view_scan(scan_id):
    scan = Scan.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        flash("Unauthorized access.")
        return redirect(url_for('index'))
    
    results = json.loads(scan.results_json)
    return render_template(
        'results.html',
        results=results,
        jd=scan.jd_text,
        jd_bias=[], # Historical scans don't re-run bias for speed
        processing_time=None
    )

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
def index():
    recent_scans = []
    if current_user.is_authenticated:
        recent_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).limit(5).all()
    return render_template('index.html', recent_scans=recent_scans)

@app.route('/process', methods=['POST'])
@login_required
def process():
    start_time = time.time()

    if 'cv_files' not in request.files or 'jd_text' not in request.form:
        flash("Please upload CVs and provide a Job Description.")
        return redirect(url_for('index'))
    
    cv_files = request.files.getlist('cv_files')
    jd_text = request.form['jd_text']
    
    if not jd_text.strip():
        flash("Job Description cannot be empty.")
        return redirect(url_for('index'))
    
    if not cv_files or cv_files[0].filename == '':
        flash("No CV files selected.")
        return redirect(url_for('index'))

    jd_skills = extract_skills(jd_text)
    jd_bias = detect_bias(jd_text)
    
    results = []
    processed_cvs_texts = []
    cv_metadata = []
    seen_cv_hashes = set()

    for file in cv_files:
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{unique_id}_{original_filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            raw_text = get_text_from_file(file_path)
            
            # --- Elite Extraction ---
            contact = extract_contact_info(raw_text)
            entities = extract_entities(raw_text)
            exp_years = estimate_experience(raw_text)
            skills = extract_skills(raw_text)

            # Robust candidate name fallback:
            # 1) Use extracted name if it's not the generic fallback
            # 2) Otherwise, derive a readable name from the original filename
            extracted_name = entities.get('name') or ""
            if extracted_name.strip() and extracted_name.strip().lower() != "unknown candidate":
                candidate_name = extracted_name.strip()
            else:
                # Derive from filename: remove extension and common suffixes like 'cv', 'resume'
                base_name, _ = os.path.splitext(original_filename)
                # Replace separators with spaces
                base_name = base_name.replace("_", " ").replace("-", " ")
                # Remove common trailing tokens
                tokens = base_name.split()
                filtered = []
                for t in tokens:
                    tl = t.lower()
                    if tl in {"cv", "resume", "profile"}:
                        continue
                    filtered.append(t)
                candidate_name = " ".join(filtered) if filtered else base_name
            
            # Preprocess for vectorization
            cleaned_cv = clean_text(raw_text)

            # --- Duplicate CV Handling (TC-4) ---
            # Use a hash of the cleaned text to skip exact duplicate CV contents
            cv_hash = hash(cleaned_cv) if cleaned_cv else None
            
            if cleaned_cv:
                if cv_hash is not None and cv_hash in seen_cv_hashes:
                    # Skip duplicate CV content
                    os.remove(file_path)
                    continue

                if cv_hash is not None:
                    seen_cv_hashes.add(cv_hash)

                processed_cvs_texts.append(cleaned_cv)
                cv_metadata.append({
                    'name': candidate_name or original_filename,
                    'contact': contact,
                    'entities': entities,
                    'experience': exp_years,
                    'skills': skills,
                    'raw_text': raw_text
                })
            
            os.remove(file_path)

    if not processed_cvs_texts:
        flash("Could not extract text from the provided files.")
        return redirect(url_for('index'))

    # --- TF-IDF Matrix ---
    language = CONFIG["app"].get("language", "english")
    cleaned_jd = clean_text(jd_text, language=language)
    cleaned_cvs_for_vec = [
        clean_text(meta["raw_text"], language=language)
        for meta in cv_metadata
    ]

    all_texts = [cleaned_jd] + cleaned_cvs_for_vec
    tfidf_matrix, _ = get_tfidf_matrix(all_texts)

    jd_vector = tfidf_matrix[0:1]
    cv_vectors = tfidf_matrix[1:]

    tfidf_scores = calculate_tfidf_similarity(jd_vector, cv_vectors)

    # --- Semantic Similarity (Elite) ---
    if CONFIG["scoring"].get("use_semantic", True):
        semantic_scores = calculate_semantic_similarity(jd_text, processed_cvs_texts)
    else:
        semantic_scores = np.zeros(len(cv_metadata))

    # --- BM25 Keyword Scoring (Hybrid) ---
    if CONFIG["scoring"].get("use_bm25", True):
        bm25_scores = calculate_bm25_scores(cleaned_jd, cleaned_cvs_for_vec)
        # Normalize BM25 scores to [0,1] to keep in line with cosine similarities
        if bm25_scores.size > 0:
            max_bm25 = bm25_scores.max()
            if max_bm25 > 0:
                bm25_scores = bm25_scores / max_bm25
    else:
        bm25_scores = np.zeros(len(cv_metadata))

    # --- Integration & Ranking ---
    weight_semantic = CONFIG["scoring"].get("weight_semantic", 0.4)
    weight_tfidf = CONFIG["scoring"].get("weight_tfidf", 0.3)
    weight_bm25 = CONFIG["scoring"].get("weight_bm25", 0.2)
    weight_skill = CONFIG["scoring"].get("weight_skill_ratio", 0.3)

    highlight_sentences_k = CONFIG["scoring"].get("top_k_highlight_sentences", 3)

    for i in range(len(cv_metadata)):
        meta = cv_metadata[i]

        # Skill Gap Analysis
        matching_skills = [s for s in meta['skills'] if s.lower() in [js.lower() for js in jd_skills]]
        missing_skills = [js for js in jd_skills if js.lower() not in [s.lower() for s in meta['skills']]]

        # Skill Match Ratio
        skill_ratio = len(matching_skills) / len(jd_skills) if len(jd_skills) > 0 else 1.0

        # Weighted Smart Score (config-driven)
        smart_score = (
            (semantic_scores[i] * weight_semantic)
            + (tfidf_scores[i] * weight_tfidf)
            + (bm25_scores[i] * weight_bm25)
            + (min(skill_ratio, 1.0) * weight_skill)
        )

        # AI Summary
        summary = generate_ai_summary(meta['name'], smart_score * 100, meta['skills'], meta['experience'])

        # Explainable snippets
        top_snippets = get_top_matching_sentences(
            jd_text,
            meta["raw_text"],
            top_k=highlight_sentences_k,
            language=language,
        )

        results.append({
            'rank': 0,  # Placeholder
            'name': meta['name'],
            'email': meta['contact']['email'],
            'score': round(smart_score * 100, 2),
            'semantic_score': round(float(semantic_scores[i]) * 100, 2),
            'tfidf_score': round(float(tfidf_scores[i]) * 100, 2),
            'experience': meta['experience'],
            'skills': meta['skills'],
            'matching_skills': matching_skills[:10],
            'missing_skills': missing_skills[:5],
            'education': meta['entities']['edu'],
            'summary': summary,
            'is_top_5': False,
            'top_snippets': top_snippets,
        })
    
    # Final Rank
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    for idx, res in enumerate(results):
        res['rank'] = idx + 1
        res['is_top_5'] = idx < 5

    # --- Save Scan to History ---
    if results:
        # Robust job title extraction (first 50 chars of JD)
        job_title = jd_text.strip().split('\n')[0][:100]
        new_scan = Scan(
            job_title=job_title,
            candidate_count=len(results),
            top_score=results[0]['score'],
            results_json=json.dumps(results),
            jd_text=jd_text,
            user_id=current_user.id
        )
        db.session.add(new_scan)
        db.session.commit()

    processing_time = time.time() - start_time

    # --- Audit Log ---
    if CONFIG["app"].get("enable_audit_log", True):
        log_dir = CONFIG["app"].get("audit_log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"audit_{uuid.uuid4().hex[:8]}.json")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "jd": jd_text,
                        "jd_skills": jd_skills,
                        "jd_bias": jd_bias,
                        "num_cv": len(results),
                        "processing_time_sec": round(processing_time, 3),
                        "candidates": [
                            {
                                "rank": r["rank"],
                                "name": r["name"],
                                "score": r["score"],
                                "semantic_score": r["semantic_score"],
                                "tfidf_score": r["tfidf_score"],
                                "experience": r["experience"],
                                "skills": r["skills"],
                            }
                            for r in results
                        ],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            # Logging failure should not break user flow
            pass

    # Export CSV
    csv_path = os.path.join(UPLOAD_FOLDER, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'name', 'score', 'experience', 'email', 'summary'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    return render_template(
        'results.html',
        results=results,
        jd=jd_text,
        jd_bias=jd_bias,
        processing_time=round(processing_time, 2) if CONFIG["ui"].get("show_processing_time", True) else None,
    )

@app.route('/download')
def download():
    csv_path = os.path.join(UPLOAD_FOLDER, 'results.csv')
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name="elite_shortlist.csv")
    return "Error: File not found", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
