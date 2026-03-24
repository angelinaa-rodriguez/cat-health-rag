from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict

def chunk_text(text: str, base_metadata: dict, chunk_size: int, overlap: int) -> list[Chunk]:
    text = " ".join(text.split())
    chunks: list[Chunk] = []
    start = 0
    i = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()

        if len(chunk) >= 200:
            md = dict(base_metadata)
            md["chunk_index"] = i
            chunk_id = f"{base_metadata['doc_id']}__{base_metadata.get('source_file','doc')}__c{i}"
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk, metadata=md))
            i += 1

        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks