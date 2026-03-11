from fastapi import FastAPI

from petmed_rag.api.schemas import ChatRequest, ChatResponse, Citation
from petmed_rag.generation.answer import generate_answer
from petmed_rag.retrieval.service import retrieve_context
from petmed_rag.safety.triage import get_disclaimer, get_safety_flag

app = FastAPI(title="PetMed RAG API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question

    retrieved = retrieve_context(question, k=4)
    context_blocks = [chunk.text for chunk in retrieved]

    answer = generate_answer(question, context_blocks)
    safety_flag = get_safety_flag(question, context_blocks)
    disclaimer = get_disclaimer(safety_flag)

    seen = set()
    citations = []

    for chunk in retrieved:
        title = chunk.metadata.get("title")
        url = chunk.metadata.get("url")
        key = (title, url)

        if key in seen:
            continue

        seen.add(key)
        citations.append(
            Citation(
                id=chunk.doc_id,
                title=title,
                url=url,
            )
        )

    return ChatResponse(
        answer=answer,
        citations=citations,
        safety_flag=safety_flag,
        disclaimer=disclaimer,
    )