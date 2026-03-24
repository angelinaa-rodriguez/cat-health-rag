def unique_sources(chunks):
    sources = []
    seen = set()

    for c in chunks:
        title = (
            c.metadata.get("title")
            or c.metadata.get("publisher")
            or c.metadata.get("file_name")
            or c.metadata.get("doc_id")
            or "Unknown source"
        )
        url = c.metadata.get("url")
        doc_id = getattr(c, "doc_id", None)

        key = (title, url)
        if key in seen:
            continue

        seen.add(key)
        sources.append(
            {
                "id": doc_id,
                "title": title,
                "url": url,
            }
        )

    return sources