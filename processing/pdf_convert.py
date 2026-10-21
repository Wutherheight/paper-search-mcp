# processing/pdf_convert.py
import httpx
import pymupdf4llm
from pathlib import Path


async def download_and_convert(pdf_url: str, output_dir: str = "./pdfs") -> str | None:
    """下载PDF并转换为Markdown"""
    if not pdf_url:
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = pdf_url.split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    pdf_path = Path(output_dir) / filename

    # 下载
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        #async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
    except Exception as e:
        print(f"下载失败 {pdf_url}: {e}")
        return None

    # 转换为Markdown
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        md_path = pdf_path.with_suffix(".md")
        md_path.write_text(md_text, encoding="utf-8")
        return md_text
    except Exception as e:
        print(f"转换失败 {pdf_path}: {e}")
        return None