# task_agent/summary_agent.py

from dotenv import load_dotenv
load_dotenv()  # 自动读取项目根目录的 .env 文件

from openai import AsyncOpenAI
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

async def summarize_paper(paper: dict, markdown_text: str) -> str:
    """生成单篇论文的结构化总结"""
    # 如果全文太长，截取前8000字符
    content = markdown_text[:8000] if markdown_text else paper.get("abstract", "")

    prompt = f"""请对以下论文生成结构化总结，包含：
1. **研究问题**：论文试图解决什么问题
2. **核心方法**：使用了什么方法/技术
3. **主要结论**：得出了什么结论
4. **与用户研究想法的关联**：这篇论文对用户的研究有什么参考价值

论文标题：{paper['title']}
论文内容：
{content}"""

    response = await get_client().chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.2
    )
    return response.choices[0].message.content


async def generate_research_report(research_idea: str, paper_summaries: list[dict]) -> str:
    """生成综合研究报告"""
    summaries_text = "\n\n---\n\n".join([
        f"### {s['title']}\n{s['summary']}" for s in paper_summaries
    ])

    prompt = f"""基于以下研究想法和论文总结，生成一份综合研究报告：

**研究想法**：{research_idea}

**各论文总结**：
{summaries_text}

请生成一份研究报告，包含：
1. **研究现状概述**：当前该领域的主要研究方向和方法
2. **关键论文对比**：各论文的方法差异和优缺点
3. **研究空白与机会**：当前研究尚未解决的问题
4. **推荐研究方向**：基于以上分析，建议的研究切入点
5. **参考论文列表**：按重要性排序"""

    response = await get_client().chat.completions.create(
        model="deepseek-flash",
        messages=[{"role": "user", "content": prompt}],
        extra_body={"thinking": {"type": "disabled"}},
        temperature=0.3
    )
    return response.choices[0].message.content