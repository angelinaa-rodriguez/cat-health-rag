from fastapi import FastAPI, HTTPException

from petmed_rag.utils.citations import unique_sources
from petmed_rag.api.schemas import ChatRequest, ChatResponse, Citation
from petmed_rag.generation.answer import generate_answer
from petmed_rag.retrieval.service import retrieve_context
from petmed_rag.safety.triage import get_disclaimer, get_safety_flag

app = FastAPI(
    title="PetMed RAG API",
    description="A retrieval-augmented API for answering cat health questions using veterinary sources.",
    version="1.0.0",
)


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.post(
        "/chat", 
        response_model=ChatResponse,
        tags=["Chat"],
        summary="Ask a cat health question",
        description="Retrieves relevant veterinary context, generates an answer, and returns citations plus safety metadata.",
)
def chat(request: ChatRequest):
    question = request.question
    chunks = retrieve_context(question, k=5)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    answer = generate_answer(question, chunks)

    safety_flag = get_safety_flag(question, [c.text for c in chunks])
    disclaimer = get_disclaimer(safety_flag)

    citations = [Citation(**source) for source in unique_sources(chunks)]
    
    return ChatResponse(
        answer=answer,
        citations=citations,
        safety_flag=safety_flag,
        disclaimer=disclaimer,
    )