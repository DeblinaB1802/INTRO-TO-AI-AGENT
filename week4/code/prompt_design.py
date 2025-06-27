import json
from sentence_transformers import SentenceTransformer, util
from domain_prompts import DOMAIN_PROMPTS

file = r"C:\Users\debli\OneDrive\Desktop\CV_PROJECT\INTRO_TO_AI_AGENTS\week4\domains.json"
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_style_conditioned_prompt(user_level : str ="high_school") -> str:
    """Generate prompt with style conditioning."""
    style_guidelines = {
        "middle_school": (
            "Use very simple language and explain concepts as if talking to a curious 12-year-old. "
            "Include relatable analogies and avoid technical terms unless explained."
        ),
        "high_school": (
            "Use simple words and a friendly tone, like you're explaining to a friend. "
            "Include fun analogies and basic real-life examples."
        ),
        "college": (
            "Use precise terms with a clear and professional tone. Provide deeper explanations and structured examples. "
            "Introduce relevant technical terms and explain them."
        ),
        "postgraduate": (
            "Assume strong foundational knowledge. Use technical language and detailed examples. "
            "You may include references to advanced theories or models if appropriate."
        ),
        "expert": (
            "Use expert-level vocabulary and domain-specific terminology. "
            "Assume fluency in core concepts and focus on advanced analysis, critique, or synthesis."
        )
    }
    style = style_guidelines.get(user_level, style_guidelines["high_school"])
    return style

def embed_domains(file: str = file) -> dict:
    """Generate embeddings for domain prototype queries"""
    with open(file, 'r') as f:
        DOMAIN_PROTOTYPES = json.load(f)

    domain_embeddings = {
        domain: model.encode(examples, convert_to_tensor=True) for domain, examples in DOMAIN_PROTOTYPES.items()
    }
    return domain_embeddings

def detect_domain(query: str, domain_embeddings: dict, threshold: int = 0.1) -> str:
    """Similarity search based domain detection"""
    query_embedding = model.encode(query, convert_to_tensor=True)
    best_score = -1
    best_domain = "default"

    for domain, domain_embeds in domain_embeddings.items():
        similarity = util.cos_sim(query_embedding, domain_embeds).max().item()
        if similarity > best_score:
            best_score = similarity
            best_domain = domain
            print(best_score)
    return best_domain if best_score >= threshold else "default"

def build_domain_specific_prompt(query: str, domain_embeddings: dict, context: list[dict], style: str) -> list[dict]:
    """Process query with style and domain specific language."""
    domain = detect_domain(query, domain_embeddings, threshold=0.6)
    domain_prompt = DOMAIN_PROMPTS[domain]
    styled_domain_prompt = f"Domain-specific instructions: {domain_prompt}. Additional instructions: {style}"
    messages = [
        {"role": "system", "content": styled_domain_prompt},
        *[{"role": msg["role"], "content": msg["content"]} for msg in context],
        {"role": "user", "content": query}
    ]
    return messages

