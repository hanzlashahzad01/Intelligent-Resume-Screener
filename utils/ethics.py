def detect_bias(text):
    """
    Checks for potentially biased language and provides neutral suggestions.
    Useful for 'Ethical AI' requirements in Europe.
    """
    bias_map = {
        "ninja": "Software Engineer / Expert",
        "rockstar": "High-performing Developer",
        "guru": "Subject Matter Expert",
        "wizard": "Specialist",
        "chairman": "Chair / Chairperson",
        "manpower": "Workforce / Personnel",
        "mankind": "Humankind / People",
        "recent graduate": "Entry-level professional",
        "digital native": "Tech-savvy professional",
        "energetic": "Motivated / Dynamic",
        "native speaker": "Fluent in [Language]",
        "top-tier university": "Relevant academic background"
    }
    
    bias_categories = {
        "gendered": ["ninja", "rockstar", "guru", "wizard", "chairman", "manpower", "mankind"],
        "ageist": ["recent graduate", "digital native", "energetic"],
        "exclusionary": ["native speaker", "top-tier university"]
    }
    
    found_bias = []
    text_lower = text.lower()
    
    for category, words in bias_categories.items():
        for word in words:
            if word in text_lower:
                found_bias.append({
                    "category": category, 
                    "word": word,
                    "suggestion": bias_map.get(word, "Neutral alternative")
                })
                
    return found_bias

def generate_ai_summary(name, score, top_skills, experience):
    """
    Generates a professional, high-impact summary of the candidate.
    """
    if score > 85:
        quality = "exceptional"
        action = "highly recommended for immediate interview"
    elif score > 70:
        quality = "strong"
        action = "well-suited for this position"
    elif score > 50:
        quality = "solid"
        action = "a viable candidate with relevant background"
    else:
        quality = "potential"
        action = "may require further technical screening"
        
    exp_text = f"{experience}+ years of relevant experience" if experience > 0 else "a fresh perspective and academic foundation"
    
    skill_highlights = ", ".join(top_skills[:3]) if top_skills else "general technical proficiency"
    
    summary = (
        f"{name} is an {quality} candidate, demonstrating {exp_text}. "
        f"They show particular strength in {skill_highlights}, making them {action}."
    )
    return summary
