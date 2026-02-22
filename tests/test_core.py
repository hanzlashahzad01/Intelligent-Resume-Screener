import pytest
from nlp.preprocess import clean_text
from utils.skills import extract_skills
from utils.ethics import detect_bias

def test_clean_text():
    text = "Hello World! Visit https://google.com for more info."
    cleaned = clean_text(text)
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "http" not in cleaned

def test_extract_skills_exact():
    text = "I am a Python and Java developer."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "Java" in skills

def test_extract_skills_fuzzy():
    # Testing fuzzy matching: "NodeJS" should match "Node.js" in SKILLS_DB
    text = "Expert in NodeJS and React"
    skills = extract_skills(text)
    assert "Node.js" in skills
    assert "React" in skills

def test_detect_bias():
    text = "We need a coding ninja for this role."
    biases = detect_bias(text)
    assert len(biases) > 0
    assert biases[0]["word"] == "ninja"
    assert "suggestion" in biases[0]

def test_no_bias():
    text = "Looking for a Software Engineer."
    biases = detect_bias(text)
    assert len(biases) == 0
