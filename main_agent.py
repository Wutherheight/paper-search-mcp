# main_agent.py

from dotenv import load_dotenv
load_dotenv()

import asyncio
import sys
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from pathlib import Path

# 如果在 Windows 上运行，显式设置事件循环策略
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def run_main_agent(user_input: str):
    # 1. 启动 MCP Server 子进程并连接
    #    - 使用 sys.executable 避免 Git Bash 找不到 python.exe 的问题
    #    - 显式传入 os.environ，确保 API Key 和环境变量能传给子进程
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=os.environ.copy()
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 2. 获取 MCP Server 提供的工具列表
            tools = await session.list_tools()
            print("可用工具：", [t.name for t in tools.tools], file=sys.stderr)

            print(f"\n研究想法：{user_input}", file=sys.stderr)
            print("正在检索论文...（首次运行需下载模型，请耐心等待）", file=sys.stderr)

            import time
            start_time = time.time()

            # 3. 调用工具
            # 如果 SDK 支持超时参数，可以显式设置。如果不支持，请通过环境变量 MCP_SERVER_REQUEST_TIMEOUT 设置。
            result = await session.call_tool(
                "search_papers",
                arguments={
                    "research_idea": user_input,
                    "max_results": 20,
                    "top_k": 3
                },
                read_timeout_seconds=600  # 设置 10 分钟超时
            )

            elapsed = time.time() - start_time
            print(f"\n检索完成，耗时 {elapsed:.2f} 秒\n", file=sys.stderr)
            print("=" * 70, file=sys.stderr)

            # 4. 展示结果
            #for content in result.content:
            #    if hasattr(content, "text"):
            #        print(content.text)
            for content in result.content:
                if hasattr(content, "text"):
                    output_text = content.text
                    print(output_text)

                    # 生成带时间戳的文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 把研究想法作为文件名的一部分（去掉不合法字符）
                    safe_idea = "".join(c for c in user_input if c.isalnum() or c in " _-")[:30]
                    filename = f"report_{timestamp}_{safe_idea}.md"

                    # 写入文件
                    output_path = Path("reports") / filename
                    output_path.parent.mkdir(exist_ok=True)
                    output_path.write_text(output_text, encoding="utf-8")
                    print(f"\n📄 报告已保存到：{output_path}", file=sys.stderr)

            return result


if __name__ == "__main__":
    research_idea = input("请输入你的研究想法：").strip()
    if not research_idea:
        research_idea = "few-shot image classification"
    asyncio.run(run_main_agent(research_idea))