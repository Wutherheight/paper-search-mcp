# processing/ranking.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def rank_by_relevance(research_idea: str, papers: list[dict]) -> list[dict]:
    """计算研究想法与每篇论文摘要的语义相似度，按相似度降序排列"""
    model = get_model()

    query_embedding = model.encode(research_idea, convert_to_tensor=True)
    abstracts = [p.get("abstract", "")[:512] for p in papers]
    paper_embeddings = model.encode(abstracts, convert_to_tensor=True,
                                    batch_size=32, show_progress_bar=False)

    scores = cos_sim(query_embedding, paper_embeddings)[0].cpu().numpy()

    for i, p in enumerate(papers):
        p["relevance_score"] = float(scores[i])

    papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    return papers