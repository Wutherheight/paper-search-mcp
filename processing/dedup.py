# processing/dedup.py
from difflib import SequenceMatcher
import re


def normalize_title(title: str) -> str:
    """标题标准化：转小写、去特殊字符、压缩空格"""
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    return ' '.join(title.split())


def title_similarity(t1: str, t2: str) -> float:
    return SequenceMatcher(None, normalize_title(t1), normalize_title(t2)).ratio()


def deduplicate(papers: list[dict]) -> list[dict]:
    """DOI优先 + 标题模糊匹配去重"""
    seen_dois = {}
    unique_papers = []

    for p in papers:
        doi = p.get("doi", "").strip().lower()
        if doi and doi in seen_dois:
            # DOI重复：合并信息（优先保留有PDF的）
            existing = seen_dois[doi]
            if not existing.get("pdf_url") and p.get("pdf_url"):
                existing["pdf_url"] = p["pdf_url"]
            continue

        # 标题模糊匹配
        is_dup = False
        for existing in unique_papers:
            if title_similarity(p["title"], existing["title"]) > 0.92:
                if not existing.get("pdf_url") and p.get("pdf_url"):
                    existing["pdf_url"] = p["pdf_url"]
                is_dup = True
                break

        if not is_dup:
            unique_papers.append(p)
            if doi:
                seen_dois[doi] = p

    return unique_papers