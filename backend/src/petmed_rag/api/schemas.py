from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "My cat has been vomiting for two days. What should I do?"
            }
        }
    }


class Citation(BaseModel):
    id: str
    title: Optional[str] = None
    url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    safety_flag: str
    disclaimer: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "Vomiting can have many causes in cats, including dietary upset, hairballs, or more serious illness. If vomiting is frequent or your cat is lethargic, seek veterinary care.",
                "citations": [
                    {
                        "id": "doc_123",
                        "title": "Feline Gastrointestinal Signs",
                        "url": "https://example.com/feline-gastro"
                    }
                ],
                "safety_flag": "normal",
                "disclaimer": "This is not a diagnosis. Contact a veterinarian if symptoms persist or worsen."
            }
        }
    }