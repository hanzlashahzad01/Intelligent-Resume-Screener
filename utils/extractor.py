import os
import re
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

def get_text_from_file(file_path):
    """
    Extracts text from PDF, DOCX or TXT files.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            text = extract_pdf_text(file_path)
            # Clean layout-induced control characters (like form feeds \f)
            text = re.sub(r'[\x0c\x00-\x08\x0b\x0e-\x1f\x7f]', '\n', text)
            return text
        elif ext == '.docx':
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_contact_info(text):
    """
    Extracts email and phone with improved precision.
    Specifically handles messy cases where email is concatenated with
    other text like: 'queries.contact0306...hanzla@gmail.comchishtian'.
    """
    if not text:
        return {"email": "N/A", "phone": "N/A"}

    raw = text

    # --- Email extraction ---
    email_result = "N/A"

    # Find the first plausible '@' and build email around it manually
    at_index = raw.find("@")
    if at_index != -1:
        # Allowed characters in local part and domain
        allowed_local = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-")
        allowed_domain = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")

        # Walk left from '@' to find start of local part
        start = at_index - 1
        while start >= 0 and raw[start] in allowed_local:
            start -= 1
        local_raw = raw[start + 1 : at_index]

        # Clean local part: remove obvious noise like 'queries', 'contact' and long digit prefixes
        # Example raw: 'queries.contact03062160564hanzlashahzadhanzlashahzad'
        alpha_tokens = re.findall(r"[A-Za-z]+", local_raw)
        noise_tokens = {"queries", "query", "contact", "phone", "email", "info"}
        filtered_tokens = [t for t in alpha_tokens if t.lower() not in noise_tokens]

        if filtered_tokens:
            # Join remaining tokens; this handles names like 'John Doe' or 'john.doe'
            local_part = "".join(filtered_tokens)
        else:
            # Fallback to raw segment
            local_part = local_raw

        # Walk right from '@' to get raw domain string
        end = at_index + 1
        while end < len(raw) and raw[end] in allowed_domain:
            end += 1
        domain_raw = raw[at_index + 1 : end]

        # Split domain_raw into domain and TLD, trimming any extra trailing letters
        if "." in domain_raw and local_part:
            first_dot = domain_raw.find(".")
            domain_name = domain_raw[:first_dot]
            tld_raw = domain_raw[first_dot + 1 :]

            # Keep only alphabetic chars in TLD and take first 2–3 letters
            tld_letters = "".join(ch for ch in tld_raw if ch.isalpha())
            if len(tld_letters) >= 2:
                tld = tld_letters[:3]  # 'com', 'net', 'org', etc.
                email_candidate = f"{local_part}@{domain_name}.{tld}".lower()

                # Final sanity check for email format
                if re.match(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}$", email_candidate):
                    email_result = email_candidate

    # --- Phone extraction ---
    phone_result = "N/A"
    phone_patterns = [
        r"(\+?\d{1,4}[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})",  # Standard format
        r"(\d{10,15})",  # Just digits (for concatenated cases)
    ]

    for pattern in phone_patterns:
        phones = re.findall(pattern, raw)
        for p in phones:
            digits = re.sub(r"\D", "", p)
            if 8 <= len(digits) <= 15:
                if len(digits) == 11 and digits.startswith("0"):
                    formatted = f"{digits[:4]}-{digits[4:7]}-{digits[7:]}"
                elif len(digits) >= 10:
                    formatted = p.strip() if len(p.strip()) < 20 else digits
                else:
                    formatted = digits
                phone_result = formatted
                break
        if phone_result != "N/A":
            break

    return {"email": email_result, "phone": phone_result}
