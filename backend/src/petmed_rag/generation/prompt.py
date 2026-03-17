def build_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n---\n\n".join(
        f"Source {i+1}:\n{block}" for i, block in enumerate(context_blocks)
    )

    return f"""
Your job is to answer the user's question using only the provided sources.

Rules:
- Use only the provided sources.
- Do not invent facts.
- Do not provide a definitive diagnosis.
- If the answer cannot be supported by the sources, say you are unsure.
- Do not copy or paste long passages from the sources.
- Write a short, clear, synthesized answer in your own words.
- Focus on the most relevant medical information first.
- If the sources mention urgent warning signs, clearly mention them.
- If the question sounds urgent or the sources describe emergency signs, recommend contacting a veterinarian promptly.
- This system provides informational guidance only and is not a substitute for professional veterinary care.

Answer format:
1. Start with a short direct answer to the question.
2. Briefly mention common possible causes if supported by the sources.
3. Mention when veterinary care is recommended if the sources support it.
4. Keep the answer concise and easy to understand.
5. Do NOT mention "Source 1", "Source 2", or refer to the retrieval process.

Question:
{question}

Sources:
{context}
""".strip()