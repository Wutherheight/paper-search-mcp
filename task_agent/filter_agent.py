# task_agent/filter_agent.py

from dotenv import load_dotenv
load_dotenv()  # 自动读取项目根目录的 .env 文件

from openai import AsyncOpenAI
import json
import os

#client = AsyncOpenAI(base_url="YOUR_API_BASE", api_key="YOUR_KEY")
#client = AsyncOpenAI(
#    base_url="https://api.deepseek.com",
#    api_key=os.environ.get("DEEPSEEK01_API_KEY")
#)
def get_client():
    """延迟初始化 DeepSeek 客户端，避免模块导入时就崩溃"""
    api_key = os.environ.get("DEEPSEEK01_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 DEEPSEEK01_API_KEY 环境变量。"
            "请在 Git Bash 中执行: export DEEPSEEK01_API_KEY='sk-...'"
        )
    return AsyncOpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key
    )

async def filter_top_k(research_idea: str, papers: list[dict], k: int = 5) -> list[dict]:
    """任务Agent: 对候选论文进行二次相关性筛选"""

    # 构造候选论文摘要列表
    candidates_text = "\n\n".join([
        f"[{i}] Title: {p['title']}\nAbstract: {p.get('abstract', 'N/A')[:400]}..."
        for i, p in enumerate(papers[:20])  # 最多送20篇给LLM
    ])

    prompt = f"""你是一个学术论文筛选专家。用户的研究想法是：

{research_idea}

以下是候选论文列表（按初步相关性排序）：

{candidates_text}

请对每篇论文从以下4个维度打分（1-10分）：
1. 相关性：论文主题与研究想法的匹配程度
2. 有用性：论文方法/结论对研究的参考价值
3. 影响力：论文的学术影响力（根据发表venue和引用推断）
4. 噪声程度：论文偏离主题的程度（分数越高越偏离）

请以JSON格式返回，格式：
{{"rankings": [{{"index": 0, "relevance": 8, "usefulness": 7, "impact": 6, "noise": 2, "reason": "简要理由"}}, ...]}}
只返回JSON，不要其他内容。"""

    response = await get_client().chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    rankings = result["rankings"]

    # 综合评分 = 相关性*0.35 + 有用性*0.35 + 影响力*0.15 - 噪声*0.15
    for r in rankings:
        r["final_score"] = (r["relevance"] * 0.35 + r["usefulness"] * 0.35
                            + r["impact"] * 0.15 - r["noise"] * 0.15)

    rankings.sort(key=lambda x: x["final_score"], reverse=True)
    top_indices = [r["index"] for r in rankings[:k]]

    top_papers = []
    for idx in top_indices:
        paper = papers[idx].copy()
        paper["agent_score"] = next(
            r["final_score"] for r in rankings if r["index"] == idx
        )
        paper["agent_reason"] = next(
            r["reason"] for r in rankings if r["index"] == idx
        )
        top_papers.append(paper)

    return top_papers