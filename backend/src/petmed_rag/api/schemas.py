from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    id: str
    title: str | None = None
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    safety_flag: str
    disclaimer: str