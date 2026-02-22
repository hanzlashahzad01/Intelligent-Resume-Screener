import re

try:
    import spacy
    try:
        # Lightweight English model for PERSON / ORG detection
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None

def extract_name(text):
    """
    Tries to find the candidate name with improved logic.
    """
    # Small window from top of CV where name + title usually appear
    top_text = text[:1200]

    # 0. Pattern: "HANZLA SHAHZAD SOFTWARE ENGINEER" style lines
    #    -> Capture name part before common role titles
    title_keywords = [
        "software engineer", "full stack developer", "full-stack developer",
        "web developer", "data scientist", "machine learning engineer",
        "backend developer", "frontend developer", "devops engineer",
        "software developer"
    ]
    for kw in title_keywords:
        # Case-insensitive match: "<Name...> <ROLE TITLE>"
        pattern = rf"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){{0,3}})\s+{kw}"
        m = re.search(pattern, top_text, flags=re.IGNORECASE)
        if m:
            candidate_name = m.group(1).strip()
            # basic sanity check: at least two tokens
            if len(candidate_name.split()) >= 2:
                return candidate_name

    # 1. Try spaCy PERSON entities first (most reliable when available)
    if nlp:
        try:
            doc = nlp(text[:2000])
            person_ents = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
            # Prefer 2–3 token names, without digits
            clean_person_ents = []
            for ent in person_ents:
                if any(ch.isdigit() for ch in ent):
                    continue
                tokens = ent.split()
                if 1 <= len(tokens) <= 4:
                    clean_person_ents.append(ent)

            if clean_person_ents:
                # Return the first reasonable PERSON entity as candidate name
                return clean_person_ents[0]
        except Exception:
            # Fall back to heuristic logic below
            pass

    # 2. Try to find name at the start, ignoring common noise
    # Expand noise filtering to prevent "AI based tools" etc. from being names
    clean_text = re.sub(r'[^a-zA-Z\s\n|]', ' ', top_text)
    lines = [l.strip() for l in re.split(r'\n|\|| {2,}', clean_text) if l.strip()]
    
    # Common headers and noise phrases to avoid
    headers = [
        "curriculum", "resume", "profile", "email", "phone", "contact", 
        "experience", "education", "skills", "summary", "tools", 
        "technologies", "projects", "languages", "about", "objective",
        "ai based tools", "technical skills", "professional summary"
    ]
    
    # Titles that are NOT names
    job_titles = ["engineer", "developer", "manager", "analyst", "specialist", "expert", "consultant", "architect", "lead"]

    # Phrases that strongly suggest this is a project / tool name, not a person
    non_name_keywords = [
        "tool", "project", "system", "application", "platform",
        "dashboard", "solution", "real time", "real-time",
        "role", "mern role"
    ]

    # If we hit these section headings, we stop searching for a name (too deep in CV)
    section_stop_keywords = [
        "work experience", "experience", "technical skills", "skills",
        "projects", "education"
    ]

    max_lines_for_name = 15
    for i, line in enumerate(lines[:max_lines_for_name]):
        line_lower = line.lower()

        # Stop if we've clearly entered a later section
        if any(kw in line_lower for kw in section_stop_keywords):
            break

        if any(x in line_lower for x in headers):
            continue
            
        words = line.split()
        # Names are usually 2-3 words, mostly capitalized
        if 2 <= len(words) <= 4:
            # Check if it looks like a name (Capitals, no digits, not a job title / project/tool)
            if all(w[0].isupper() for w in words if len(w) > 0) and not any(ch.isdigit() for ch in line):
                if not any(title in line_lower for title in job_titles) and not any(
                    k in line_lower for k in non_name_keywords
                ):
                    # Check if it contains common name noise
                    if not any(noise in line_lower for noise in ["using", "built", "work", "design"]):
                        return line
        
        # Look for "Name: John Doe" pattern
        name_match = re.search(r'Name\s*:\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', line)
        if name_match:
            return name_match.group(1)
            
    # Final fallback: First non-header line that isn't too long and doesn't look like a role/project
    for line in lines[:5]:
        line_lower = line.lower()
        if (
            len(line.split()) <= 4
            and not any(h in line_lower for h in headers)
            and not any(k in line_lower for k in non_name_keywords)
            and not any(ch.isdigit() for ch in line)
        ):
            return line
            
    return "Unknown Candidate"

def extract_entities(text):
    """
    Robust extraction for chaotic text.
    """
    entities = {"name": extract_name(text), "orgs": [], "dates": [], "edu": []}
    
    # 1. Broad Education Search (Regex based)
    # Expand keywords to capture more variations
    edu_keywords = [
        # Degrees / programs
        "Bachelor", "Bachelors", "Master", "Masters", "PhD", "Doctorate",
        "BSc", "MSc", "BCA", "MCA", "BBA", "MBA",
        "B.Tech", "M.Tech", "B.E.", "M.E.", "B.S.", "M.S.",
        "B.Com", "M.Com", "Diploma",
        # Institutions
        "University", "College", "School", "Institute", "Academy",
        # Generic education words
        "Degree", "Graduation", "Graduate", "Post Graduate", "Higher Secondary",
        "Intermediate", "Matric"
    ]
    
    # 1a. Prefer explicit EDUCATION / AUSBILDUNG sections when present
    for heading in ["education", "ausbildung"]:
        pattern = rf"(?i){heading}\s*[:\-]?\s*(.*?)(\n\s*\n|$)"
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            block = match.group(1)
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                if any(k.lower() in line.lower() for k in edu_keywords):
                    entities["edu"].append(line)

    # 1b. Fallback: Multi-line search based on edu keywords anywhere in text
    # Use broader split to handle bullet points and chaotic layouts
    chunks = re.split(r'\n|\||•|\*| {3,}| - |: ', text)
    for chunk in chunks:
        chunk = chunk.strip()
        if any(k.lower() in chunk.lower() for k in edu_keywords):
            # Better length constraints: Education lines can be short (e.g. "B.Tech CSE")
            if 3 < len(chunk) < 150:
                # Filter out obvious project noise or experience noise
                noise = ["using", "built", "developed", "worked", "responsibility", "skills:", "technologies:"]
                if not any(n in chunk.lower() for n in noise):
                    # Clean up the chunk (remove things like "Education:")
                    clean_chunk = re.sub(r'^(Education|Qualifications|Academia)\s*:\s*', '', chunk, flags=re.IGNORECASE)
                    entities["edu"].append(clean_chunk)
    
    # 2. Date/Year Search
    # Use non-capturing group so we get full years like 2018, 2020 instead of just 19/20
    year_matches = re.findall(r'\b(?:19|20)\d{2}\b', text)
    entities["dates"] = sorted(list(set(year_matches)), reverse=True)
    
    # 3. Org / institution search (spaCy if available)
    if nlp:
        try:
            doc = nlp(text[:8000])
            entities["orgs"] = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))[:8]
        except Exception:
            pass

    # Better cleaning for education
    final_edu = []
    seen = set()
    for item in entities["edu"]:
        # Normalize spaces and add missing spaces around years and capital letters
        clean_item = re.sub(r'\s+', ' ', item).strip()
        clean_item = re.sub(r'(\D)(\d{4})', r'\1 \2', clean_item)
        clean_item = re.sub(r'(\d{4})([A-Z])', r'\1 \2', clean_item)

        # Basic normalization to avoid duplicates
        norm = clean_item.lower().replace(".", "").replace(" ", "").strip()
        if norm not in seen and len(norm) > 1:
            final_edu.append(clean_item)
            seen.add(norm)
            
    entities["edu"] = final_edu[:5]
    
    return entities

def estimate_experience(text):
    """
    Robust experience estimation.
    """
    # Look for patterns like 'X years of experience', 'X+ years exp'
    exp_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs)\s*(?:of)?\s*(?:experience|exp)?', text.lower())
    if exp_matches:
        return max([int(x) for x in exp_matches])
        
    # Look for any digit followed by 'years' in general
    more_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs)', text.lower())
    if more_matches:
        # Filter for reasonable work experience numbers
        valid = [int(x) for x in more_matches if int(x) < 30]
        if valid: return max(valid)
        
    # Date range fallback: infer career span from years mentioned in CV
    years = sorted([int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b', text)])
    if len(years) >= 2:
        current_year = 2026
        # Assume career started after 2000 for this context
        career = [y for y in years if y > 2000 and y <= current_year]
        if len(career) >= 2:
            return max(career) - min(career)
            
    return 0
