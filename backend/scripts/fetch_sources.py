from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

RAW_DIR = Path("data/raw")
META_DIR = Path("data/raw/_meta")
SOURCES_FILE = Path("data/sources.yaml")


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80]


def stable_filename(source_id: str, url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{slugify(source_id)}__{h}.html"


def load_sources() -> list[dict]:
    data = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    return data.get("sources", [])


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_sources()
    if not sources:
        raise SystemExit("No sources found in data/sources.yaml")

    headers = {
        "User-Agent": "petmed-rag/0.1 (+portfolio project; respectful crawler)"
    }

    with httpx.Client(timeout=30.0, headers=headers, follow_redirects=True) as client:
        for s in sources:
            url = s["url"]
            source_id = s["id"]

            fname = stable_filename(source_id, url)
            out_html = RAW_DIR / fname
            out_meta = META_DIR / (fname.replace(".html", ".json"))

            print(f"Fetching: {source_id} -> {url}")
            r = client.get(url)
            r.raise_for_status()

            out_html.write_bytes(r.content)

            meta = {
                "id": source_id,
                "url": url,
                "title": s.get("title"),
                "publisher": s.get("publisher"),
                "species": s.get("species", []),
                "topic_tags": s.get("topic_tags", []),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": r.status_code,
                "final_url": str(r.url),
                "content_type": r.headers.get("content-type"),
                "hostname": urlparse(str(r.url)).hostname,
                "raw_filename": out_html.name,
            }
            out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nDone. Raw HTML saved to data/raw and metadata to data/raw/_meta")


if __name__ == "__main__":
    main()
