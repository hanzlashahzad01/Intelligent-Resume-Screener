import re
from thefuzz import process, fuzz

# Predefined skills list (can be expanded)
SKILLS_DB = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "React", "Angular", "Vue", "Node.js", "Express",
    "Flask", "Django", "FastAPI", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Scikit-Learn",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Data Science", "Data Analysis", "Tableau", "Power BI",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Git", "GitHub", "GitLab", "Jenkins", "Terraform",
    "Linux", "Bash", "Shell", "Agile", "Scrum", "API", "REST", "GraphQL", "Microservices", "Unit Testing",
    "HTML", "CSS", "Sass", "Bootstrap", "Tailwind CSS", "Redux", "PHP", "Laravel", "Ruby", "Rails",
    "Go", "Rust", "Swift", "Kotlin", "Flutter", "React Native", "Firebase", "Heroku", "Netlify", "Vercel"
]

def extract_skills(text, threshold=90):
    """
    Extracts skills from text based on the predefined SKILLS_DB.
    Uses a combination of regex (exact) and fuzzy matching (for variations like NodeJS vs Node.js).
    """
    found_skills = set()
    text_clean = text.lower()
    
    # 1. Exact/Regex Match (Fast)
    for skill in SKILLS_DB:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_clean):
            found_skills.add(skill)
    
    # 2. Fuzzy Matching for variations (e.g., "NodeJS" matching "Node.js")
    # We split the text into tokens to check for fuzzy matches against SKILLS_DB
    tokens = re.findall(r'\b\w+[\.\-#]*\w*\b', text_clean)
    for token in tokens:
        if len(token) < 3: continue # Skip very short tokens
        
        # Check if token matches any skill in DB with high similarity
        best_match, score = process.extractOne(token, SKILLS_DB, scorer=fuzz.token_set_ratio)
        if score >= threshold:
            found_skills.add(best_match)
            
    return sorted(list(found_skills))

def highlight_skills(text, skills):
    """
    Returns a list of skills found and their count.
    """
    found = extract_skills(text)
    return found, len(found)
