# Paper Search MCP Agent

基于 MCP 的多源论文检索与总结智能体。
声明：代码使用deepseek网页对话辅助生成。

## 功能
- 从 arXiv 检索论文
- Sentence Transformers 语义相关性排序
- 任务 Agent 二次筛选 Top-k
- PDF 转 Markdown + LLM 生成单篇总结和综合报告

## 快速开始
1. 克隆项目并创建虚拟环境
2. `pip install -r requirements.txt`
3. 复制 `.env.example` 为 `.env`，填入你的 API Key
4. `python main_agent.py`

## 环境变量
- `DEEPSEEK01_API_KEY`: DeepSeek API Key
- `HF_ENDPOINT`: Hugging Face 镜像地址

