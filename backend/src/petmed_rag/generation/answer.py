from openai import OpenAI
from petmed_rag.config import settings
from petmed_rag.generation.prompt import build_prompt

client = OpenAI(api_key=settings.openai_api_key)

def generate_answer(question: str, chunks) -> str:
    context_blocks = [c.text for c in chunks]

    prompt = build_prompt(question, context_blocks)

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a cautious cat health information assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    return resp.choices[0].message.content.strip()