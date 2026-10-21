# server.py
"""
MCP Server: 多源论文检索与总结服务
对外暴露一个工具 search_papers，内部编排所有子模块完成完整流程。
"""

# server.py 最顶部
from dotenv import load_dotenv
load_dotenv()  # 自动读取项目根目录的 .env 文件

# 必须在 import FastMCP 之前
import asyncio
import sys
import logging
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

#import asyncio
import time
import json
import traceback
from typing import Any

#from mcp.server.fastmcp import FastMCP # SDK 1.x 版本写法
from fastmcp import FastMCP
#from mcp.server.mcpserver import MCPServer as FastMCP # 适配 SDK 2.x，需要注意有些API名字要修改

# ---------- 1. 导入各模块的函数 ----------
from sources.arxiv_source import search_arxiv
from sources.s2_source import search_semantic_scholar
from processing.dedup import deduplicate
from processing.ranking import rank_by_relevance
from processing.pdf_convert import download_and_convert
from task_agent.filter_agent import filter_top_k
from task_agent.summary_agent import summarize_paper, generate_research_report


# ---------- 2. 创建 MCP 实例 ----------
mcp = FastMCP("PaperSearchServer")
#mcp = FastMCP("PaperSearchServer", request_timeout=600) # 可能需要特定版本


# ---------- 2.5 log() 日志函数 ----------
LOG_FILE = Path(__file__).parent / "server.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
    force=True,
)
_logger = logging.getLogger("PaperSearchServer")

def log(msg: str, level: str = "info"):
    """同时输出到 stderr 和 server.log 文件"""
    print(msg, file=sys.stderr, flush=True)  # 加 flush 确保不缓冲
    getattr(_logger, level)(msg)

#def log(msg: str):
#    """统一日志输出到 stderr，避免污染 MCP 的 stdout 协议通道"""
#    print(msg, file=sys.stderr, flush=True)


# ---------- 3. 内部辅助函数（编排逻辑） ----------
'''
async def _multi_source_search(query: str, max_results: int) -> list[dict]:
    """步骤1: 并发调用两个数据源，返回合并后的原始列表"""
    ########### 串行版 ##############
    #arxiv_papers = await search_arxiv(query, max_results)
    #s2_papers = await search_semantic_scholar(query, max_results)
    #results = arxiv_papers + s2_papers
    ########### 并发版 ##############
    arxiv_task = asyncio.create_task(search_arxiv(query, max_results))
    s2_task = asyncio.create_task(search_semantic_scholar(query, max_results))
    # return_exceptions=True 保证一个源挂掉不影响另一个
    results = await asyncio.gather(arxiv_task, s2_task, return_exceptions=True)
    ############################################

    all_papers = []
    for name, res in zip(["arxiv", "semantic_scholar"], results):
        if isinstance(res, Exception):
            log(f"[WARN] {name} 检索失败: {res}")
            continue
        all_papers.extend(res)

    log(f"[INFO] 多源检索完成，原始论文数: {len(all_papers)}")
    return all_papers
'''
async def _multi_source_search(query: str, max_results: int) -> list[dict]:
    # 先只跑 arXiv，确认整条链路能通
    try:
        arxiv_papers = await search_arxiv(query, max_results)
        log(f"[INFO] arXiv 原始返回 {len(arxiv_papers)} 篇")
    except Exception as e:
        log(f"[ERROR] arXiv 失败: {e}")
        arxiv_papers = []
    all_papers = list(arxiv_papers)

    # 等 arXiv 跑通后再启用 S2
    #try:
    #    s2_papers = await search_semantic_scholar(query, max_results)
    #    all_papers.extend(s2_papers)
    #except Exception as e:
    #    log(f"[WARN] S2 跳过: {e}")

    log(f"[INFO] 多源检索完成，原始论文数: {len(all_papers)}")
    return all_papers

async def _process_papers(research_idea: str,
                          raw_papers: list[dict],
                          top_k: int) -> tuple[list[dict], dict]:
    """
    步骤2+3+4: 去重 → 排序 → 任务Agent二次筛选
    返回 (top_papers, stats)
    """
    stats: dict[str, Any] = {"raw": len(raw_papers)}

    # ---- 步骤2: 去重 ----
    unique_papers = deduplicate(raw_papers)
    stats["after_dedup"] = len(unique_papers)

    # ---- 步骤3: 相关性排序（Sentence Transformers） ----
    ranked = rank_by_relevance(research_idea, unique_papers)

    # ---- 步骤4: 任务Agent 二次筛选 Top-k ----
    top_papers = await filter_top_k(research_idea, ranked, k=top_k)
    stats["top_k"] = len(top_papers)

    return top_papers, stats


async def _download_and_summarize(research_idea: str,
                                  top_papers: list[dict]) -> tuple[list[dict], str]:
    """
    步骤5(前半): 下载PDF → 转Markdown → 单篇总结
    返回 (带summary的论文列表, 综合研究报告)
    """
    # ---- 并发下载+转换所有PDF ----
    download_tasks = [
        asyncio.create_task(download_and_convert(p.get("pdf_url", "")))
        for p in top_papers
    ]
    markdown_list = await asyncio.gather(*download_tasks, return_exceptions=True)

    # ---- 并发生成单篇总结 ----
    summary_tasks = []
    for paper, md in zip(top_papers, markdown_list):
        if isinstance(md, Exception):
            md = None
        summary_tasks.append(
            asyncio.create_task(summarize_paper(paper, md or ""))
        )
    summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)

    # ---- 组装结果 ----
    enriched_papers = []
    for paper, md, summ in zip(top_papers, markdown_list, summaries):
        paper = paper.copy()
        paper["has_fulltext"] = isinstance(md, str) and bool(md)
        paper["summary"] = summ if isinstance(summ, str) else "（总结生成失败）"
        enriched_papers.append(paper)

    # ---- 生成综合研究报告 ----
    try:
        report = await generate_research_report(research_idea, enriched_papers)
    except Exception as e:
        log(f"[WARN] 综合报告生成失败: {e}")
        report = "（综合研究报告生成失败）"

    return enriched_papers, report


def _format_result(research_idea: str,
                   papers: list[dict],
                   report: str,
                   stats: dict,
                   elapsed: float) -> str:
    """把最终结果格式化为对LLM友好的文本（Markdown）"""
    lines = []
    lines.append(f"# 论文检索与研究报告\n")
    lines.append(f"**研究想法**：{research_idea}\n")
    lines.append(f"**耗时**：{elapsed:.2f} 秒")
    lines.append(f"**统计**：原始 {stats['raw']} 篇 → 去重后 "
                 f"{stats['after_dedup']} 篇 → Top-{stats['top_k']}\n")

    # ---- Top-k 列表 ----
    lines.append("## Top-k 论文列表\n")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        lines.append(
            f"### {i}. {p['title']}\n"
            f"- **作者**：{authors}\n"
            f"- **来源**：{p.get('source', 'N/A')} | "
            f"**发表**：{p.get('published', 'N/A')}\n"
            f"- **PDF**：{p.get('pdf_url') or '无公开全文'}\n"
            f"- **初步相关性**：{p.get('relevance_score', 0):.4f} | "
            f"**Agent综合评分**：{p.get('agent_score', 0):.2f}\n"
            f"- **Agent筛选理由**：{p.get('agent_reason', 'N/A')}\n"
        )

    # ---- 单篇总结 ----
    lines.append("\n## 单篇论文总结\n")
    for i, p in enumerate(papers, 1):
        lines.append(f"### {i}. {p['title']}\n")
        lines.append(p.get("summary", "（无总结）"))
        lines.append("")

    # ---- 综合报告 ----
    lines.append("\n## 综合研究报告\n")
    lines.append(report)

    return "\n".join(lines)


# ---------- 4. 对外暴露的 MCP 工具 ----------

@mcp.tool()
async def search_papers(research_idea: str = "few-shot image classification",
                        max_results: int = 30,
                        top_k: int = 5) -> str:
    """根据用户的研究想法，从多个公开学术来源检索论文，去重排序后由任务Agent
    筛选出最相关的Top-k篇，下载全文并生成单篇总结和综合研究报告。

    Args:
        research_idea: 用户的研究想法或问题描述（自然语言）
        max_results: 每个数据源检索的最大论文数（默认30）
        top_k: 最终返回的论文数量（默认5）

    Returns:
        Markdown 格式的完整结果：Top-k论文列表 + 单篇总结 + 综合研究报告
    """
    start = time.time()
    log(f"\n{'='*60}\n[INFO] 收到检索请求: {research_idea}\n{'='*60}")
    log(f"[T+0.0s] 开始检索: {research_idea}")

    try:
        # ============ 步骤1: 多源检索 ============
        raw_papers = await _multi_source_search(research_idea, max_results)
        if not raw_papers:
            return "❌ 未从任何数据源检索到论文，请检查网络或更换关键词。"
        log(f"[T+{time.time() - start:.1f}s] 检索完成，{len(raw_papers)} 篇")

        # ============ 步骤2+3+4: 去重 → 排序 → 任务Agent筛选 ============
        top_papers, stats = await _process_papers(
            research_idea, raw_papers, top_k
        )
        if not top_papers:
            return "❌ 去重/筛选后无有效论文。"
        log(f"[T+{time.time() - start:.1f}s] 去重+排序+筛选完成，Top-{len(top_papers)}")

        # ============ 步骤5: 下载全文 + 生成总结/报告 ============
        enriched_papers, report = await _download_and_summarize(
            research_idea, top_papers
        )
        log(f"[T+{time.time() - start:.1f}s] 下载+总结+报告完成")

        elapsed = time.time() - start
        log(f"[INFO] 全流程完成，总耗时 {elapsed:.2f}s")

        return _format_result(research_idea, enriched_papers, report,
                              stats, elapsed)

    except Exception as e:
        traceback.print_exc()
        return f"❌ 处理失败: {type(e).__name__}: {e}"


# 可选：暴露一个轻量级工具，只做检索不做总结，方便调试
@mcp.tool()
async def search_only(research_idea: str, max_results: int = 20) -> str:
    """只做多源检索+去重+排序，不下载全文、不做总结。用于快速测试检索质量。

    Args:
        research_idea: 研究想法
        max_results: 每个源检索数量
    """
    raw = await _multi_source_search(research_idea, max_results)
    unique = deduplicate(raw)
    ranked = rank_by_relevance(research_idea, unique)
    top = ranked[:10]
    return json.dumps(
        [{"title": p["title"], "score": p.get("relevance_score", 0),
          "source": p.get("source"), "pdf": bool(p.get("pdf_url"))}
         for p in top],
        ensure_ascii=False, indent=2
    )


# ---------- 5. 启动 ----------

if __name__ == "__main__":
    # stdio 传输：主Agent会以子进程方式启动本文件
    mcp.run(transport="stdio")