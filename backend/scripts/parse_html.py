from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from readability import Document

RAW_DIR = Path("data/raw")
RAW_META_DIR = Path("data/raw/_meta")

OUT_DIR = Path("data/processed")
OUT_META_DIR = Path("data/processed/_meta")

def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def html_to_clean_text(html: str) -> tuple[str, str | None]:
    """
    Returns: (clean_text, extracted_title)
    """
    doc = Document(html)
    title = doc.short_title() or None
    main_html = doc.summary(html_partial=True)

    soup = BeautifulSoup(main_html, "lxml")
    # Remove obvious junk
    for tag in soup(["script", "style", "noscript", "svg", "img", "form", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text("\n")
    return normalize_whitespace(text), title

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_META_DIR.mkdir(parents=True, exist_ok=True)

    html_files = sorted([p for p in RAW_DIR.glob("*.html") if p.is_file()])
    if not html_files:
        raise SystemExit("No HTML files found in data/raw. Run fetch_sources.py first.")

    processed_count = 0

    for html_path in html_files:
        meta_path = RAW_META_DIR / html_path.name.replace(".html", ".json")
        if not meta_path.exists():
            print(f"⚠️  Missing meta for {html_path.name}, skipping.")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        html = html_path.read_text(encoding="utf-8", errors="ignore")

        clean_text, extracted_title = html_to_clean_text(html)
        print(f"\n--- {html_path.name} ---")
        print(f"title: {extracted_title}")
        print(clean_text[:500])
        print("--- end preview ---\n")

        if len(clean_text) < 400:
            print(f"⚠️  Extracted text too short for {html_path.name} ({len(clean_text)} chars), skipping.")
            continue

        out_txt = OUT_DIR / html_path.name.replace(".html", ".txt")
        out_txt.write_text(clean_text, encoding="utf-8")

        # enrich meta for downstream citations
        processed_meta = dict(meta)
        processed_meta["extracted_title"] = extracted_title
        processed_meta["processed_filename"] = out_txt.name
        processed_meta["char_count"] = len(clean_text)

        out_meta = OUT_META_DIR / meta_path.name
        out_meta.write_text(json.dumps(processed_meta, indent=2), encoding="utf-8")

        processed_count += 1
        print(f"✅ Parsed {html_path.name} -> {out_txt.name}")

    print(f"\nDone. Parsed {processed_count} documents into data/processed/")

if __name__ == "__main__":
    main()
