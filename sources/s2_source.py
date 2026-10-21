# sources/s2_source.py
import httpx


async def search_semantic_scholar(query: str, max_results: int = 30) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,abstract,authors,externalIds,openAccessPdf,year,venue"
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    papers = []
    for p in data.get("data", []):
        papers.append({
            "title": p.get("title", ""),
            "abstract": p.get("abstract", "") or "",
            "authors": [a["name"] for a in p.get("authors", [])],
            "doi": (p.get("externalIds") or {}).get("DOI", ""),
            "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
            "source": "semantic_scholar",
            "published": str(p.get("year", "")),
            "venue": p.get("venue", "")
        })
    return papers