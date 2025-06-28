import json

def check_ethical_compliance(text, query = True):
    """Check response for ethical issues."""
    text_lower = text.lower()
    is_compliant = True
    msg = text
    with open(r"week4\sensitive_terms.json", 'r') as f:
        SENSITIVE_TERMS_BY_CATEGORY = json.load(f)
    for topic, terms in SENSITIVE_TERMS_BY_CATEGORY.items():
        for term in terms:
            if term in text_lower:
                is_compliant = False
                if query:
                    msg = f"""⚠️ **Caution**: Your query includes a potentially sensitive term *'{term}'* related to *{topic.replace('_',' ')}*.
                    To maintain a respectful and inclusive learning environment, Study Buddy avoids promoting such language. 
                    Please consider rephrasing. 😊
                    """
                else:
                    msg = f"""⚠️ **Notice**: The generated response includes the term *'{term}'* related to *{topic.replace('_',' ')}*, which may be inappropriate or sensitive.

                    Study Buddy does not support or promote such content and strives to maintain a safe and respectful learning environment. 😊

                    Answer: {text}
                    """
        return is_compliant, msg
    return is_compliant, msg