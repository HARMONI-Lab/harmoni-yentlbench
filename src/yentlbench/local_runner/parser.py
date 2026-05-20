import re
import json

def parse_esi(response_text: str) -> float | None:
    """
    Extracts an ESI score (1-5) from free-text LLM responses.
    Returns the score as a float, or None if unparseable.
    """
    if not response_text:
        return None
        
    text = str(response_text).strip()
    
    # 1. Try JSON parsing (since prompt requested JSON)
    try:
        # Strip markdown json block if present
        clean_text = re.sub(r'^```json\s*|\s*```$', '', text).strip()
        data = json.loads(clean_text)
        if 'score' in data:
            score = float(data['score'])
            if 1 <= score <= 5:
                return score
    except Exception:
        pass

    # 2. Try JSON-like substring match
    match = re.search(r'(?i)"score"\s*:\s*([1-5])', text)
    if match:
        return float(match.group(1))

    # 3. Try standard ESI/Level phrases
    # Handles: "ESI 3", "ESI level 2", "Level 3", "Triage Level: 3", "Score: 4"
    match = re.search(r'(?i)(?:esi(?: level)?|triage(?: level)?|level|score)\s*:?\s*([1-5])', text)
    if match:
        return float(match.group(1))

    # 4. Try exactly just the number (possibly surrounded by whitespace/punctuation)
    match = re.search(r'^\W*([1-5])\W*$', text)
    if match:
        return float(match.group(1))

    # 5. Last resort fallback: first isolated digit between 1 and 5
    match = re.search(r'\b([1-5])\b', text)
    if match:
        return float(match.group(1))

    return None
