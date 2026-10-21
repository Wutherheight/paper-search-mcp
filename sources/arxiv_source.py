# sources/arxiv_source.py
import arxiv

async def search_arxiv(query: str, max_results: int = 30) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for r in client.results(search):
        papers.append({
            "title": r.title,
            "abstract": r.summary,
            "authors": [a.name for a in r.authors],
            "doi": r.doi or "",
            "pdf_url": r.pdf_url,
            "source": "arxiv",
            "published": str(r.published),
            "arxiv_id": r.entry_id.split("/")[-1]
        })
    return papers