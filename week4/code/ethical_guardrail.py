SENSITIVE_TERMS_BY_CATEGORY = {
    "hate_speech_and_bias": [
        "racist", "racism", "sexism", "sexist", "homophobic", "transphobic", "misogynistic",
        "xenophobic", "anti-semitic", "discriminate", "slur",
        "superior race", "inferior", "hate group", "hate", "offensive", "bigot"
    ],

    "violence_and_criminal_activity": [
        "violence", "kill", "murder", "assault", "rape", "molest", "lynching",
        "exterminate", "genocide", "terrorism", "bomb", "gun", "knife",
        "abduction", "shooting", "stabbing", "arson", "slavery"
    ],

    "harassment_and_abuse": [
        "bully", "harass", "abuse", "gaslight", "manipulate", "intimidate",
        "stalking", "threaten", "retaliate", "toxic behavior", "cyberbullying"
    ],

    "mental_health_and_self_harm": [
        "suicide", "self-harm", "depression", "anxiety", "bipolar", "anorexia",
        "eating disorder", "cutting", "overdose", "panic attack",
        "mental breakdown", "burnout", "trauma", "ptsd", "hallucinate"
    ],

    "substance_use_and_addiction": [
        "drug use", "alcohol", "overdose", "addiction", "opioid", "cocaine",
        "marijuana", "lsd", "mdma", "narcotics", "prescription abuse"
    ],

    "explicit_and_inappropriate_content": [
        "sex", "porn", "nude", "nsfw", "incest", "masturbate", "strip",
        "explicit", "erotic", "fetish", "intercourse", "sexually explicit"
    ],

    "criminal_justice_and_sensitive_legal": [
        "incarceration", "death penalty", "prison abuse", "child custody",
        "probation", "felony", "forensics", "domestic violence"
    ],

    "sensitive_societal_issues": [
        "abortion", "miscarriage", "euthanasia", "homelessness",
        "war crimes", "civil unrest", "refugee crisis", "riot",
        "trafficking", "poverty", "genital mutilation"
    ],
    "potential_bias": ["better than", "smarter race", "women are weaker", "boys don't cry",
        "all men are aggressive", "inferior intelligence", "some cultures are backwards",
        "developed vs undeveloped", "third world", "white people are more successful",
        "poverty is a choice", "immigrants take our jobs"
    ]

}

def check_ethical_compliance(text, query = True):
    """Check response for ethical issues."""
    text_lower = text.lower()
    is_compliant = True
    msg = text
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
