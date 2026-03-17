from pathlib import Path
import hashlib
import json
import httpx
import yaml

RAW_DIR = Path("data/raw")
RAW_META_DIR = RAW_DIR / "_meta"


def slugify_id(source_id: str, url: str) -> str:
    short = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    return f"{source_id}__{short}"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_META_DIR.mkdir(parents=True, exist_ok=True)

    with open("data/sources.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sources = config["sources"]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with httpx.Client(headers=headers, follow_redirects=True, timeout=20.0) as client:
        for src in sources:
            try:
                print(f"Fetching: {src['id']} -> {src['url']}")
                r = client.get(src["url"])
                r.raise_for_status()

                stem = slugify_id(src["id"], src["url"])
                html_path = RAW_DIR / f"{stem}.html"
                meta_path = RAW_META_DIR / f"{stem}.json"

                html_path.write_text(r.text, encoding="utf-8")
                meta_path.write_text(json.dumps(src, indent=2), encoding="utf-8")

                print(f"Saved: {html_path}")

            except Exception as e:
                print(f"Skipping {src['id']}: {e}")
                continue


if __name__ == "__main__":
    main()