URGENT_KEYWORDS = [
    "not breathing",
    "can't breathe",
    "seizure",
    "blood",
    "collapsed",
    "unresponsive",
    "poison",
    "toxin",
    "blood",
    "coffee grounds",
    "difficulty breathing",
    "lethargy",
    "multiple vomiting episodes",
    "blockage",
    "severe pain",
    "refuses to eat",
    "refuses to drink",
]

def get_safety_flag(question: str, context_blocks: list[str]) -> str:
    text = " ".join([question] + context_blocks).lower()
    for term in URGENT_KEYWORDS:
        if term in text:
            return "urgent"
    return "normal"


def get_disclaimer(flag: str) -> str:
    if flag == "urgent":
        return "This may be urgent. Please contact a veterinarian or emergency vet as soon as possible."
    return "This is informational only and is not a substitute for veterinary care."