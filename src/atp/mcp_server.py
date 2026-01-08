from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from atp.rag.rag_core import (
    add_textfile_to_vectorstore,
    answer_query,
    build_vectorstore_from_pdfs,
)
from atp.web.scrape import scrape_url
from atp.web.search import search_urls

# Lưu ý: nếu chạy transport="stdio" thì tuyệt đối không print ra stdout
# => dùng logging ra stderr để tránh corrupt JSON-RPC. :contentReference[oaicite:2]{index=2}
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(message)s",
)

DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_DOCS_DIR = Path("docs")

# Stateless + JSON response là khuyến nghị cho streamable-http. :contentReference[oaicite:3]{index=3}
mcp = FastMCP("ATP Tool Server", stateless_http=True, json_response=True)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _web_where(url: str) -> dict:
    # Chroma where cần 1 operator => dùng $and
    return {"$and": [{"source_type": "web"}, {"url": url}]}


@mcp.tool()
def atp_web_search(query: str, limit: int = 5, allowed_domain: Optional[str] = None) -> list[str]:
    """
    Tìm URL theo keyword.
    - allowed_domain: ví dụ "viblo.asia" (1 domain)
    """
    allowed = [allowed_domain] if allowed_domain else None
    return search_urls(query=query, limit=limit, allowed_domains=allowed)


@mcp.tool()
async def atp_web_scrape(
    url: str,
    allowed_domain: Optional[str] = None,
    content_selector: Optional[str] = None,
    headless: bool = True,
    timeout_ms: int = 30000,
    out_dir: str = str(DEFAULT_OUTPUTS_DIR),
) -> dict:
    """
    Scrape URL (Playwright) -> lưu outputs/page.html + outputs/page.txt
    Trả về preview text + đường dẫn file đã lưu.
    """
    out = Path(out_dir)
    _ensure_dir(out)

    allowed = [allowed_domain] if allowed_domain else None
    r = await scrape_url(
        url,
        allowed_domains=allowed,
        headless=headless,
        timeout_ms=timeout_ms,
        content_selector=content_selector,
    )

    html_path = out / "page.html"
    txt_path = out / "page.txt"
    html_path.write_text(r.html, encoding="utf-8")
    txt_path.write_text(r.text, encoding="utf-8")

    preview = (r.text or "")[:2000]
    return {
        "url": url,
        "saved_html": str(html_path),
        "saved_text": str(txt_path),
        "text_len": len(r.text or ""),
        "text_preview": preview,
    }


@mcp.tool()
def atp_web_index(
    url: str,
    text_path: str = str(DEFAULT_OUTPUTS_DIR / "page.txt"),
    chroma_dir: str = str(DEFAULT_CHROMA_DIR),
    embed_model: str = "embeddinggemma",
) -> dict:
    """
    Index outputs/page.txt vào Chroma, gắn metadata source_type=web, url=...
    """
    tp = Path(text_path)
    if not tp.exists():
        return {"ok": False, "error": f"Không thấy file text_path: {text_path}"}

    cd = Path(chroma_dir)
    _ensure_dir(cd)

    added = add_textfile_to_vectorstore(
        text_path=tp,
        persist_dir=cd,
        embed_model=embed_model,
        metadata={"source_type": "web", "url": url},
    )
    return {"ok": True, "added_chunks": added, "chroma_dir": str(cd), "url": url}


@mcp.tool()
def atp_rag_ingest(
    docs_dir: str = str(DEFAULT_DOCS_DIR),
    chroma_dir: str = str(DEFAULT_CHROMA_DIR),
    embed_model: str = "embeddinggemma",
) -> dict:
    """
    Ingest toàn bộ PDF trong docs_dir -> Chroma
    """
    dd = Path(docs_dir)
    pdfs = sorted(dd.glob("*.pdf"))
    if not pdfs:
        return {"ok": False, "error": f"Không thấy PDF trong {docs_dir}"}

    cd = Path(chroma_dir)
    _ensure_dir(cd)

    n = build_vectorstore_from_pdfs(pdfs, cd, embed_model=embed_model)
    return {"ok": True, "indexed_chunks": n, "pdf_count": len(pdfs), "chroma_dir": str(cd)}


@mcp.tool()
def atp_rag_query(
    question: str,
    chroma_dir: str = str(DEFAULT_CHROMA_DIR),
    embed_model: str = "embeddinggemma",
    chat_model: str = "qwen3:1.7b",
    top_k: int = 4,
    url: Optional[str] = None,
) -> dict:
    """
    Hỏi đáp RAG. Nếu có url => lọc retrieval theo đúng url (không lẫn nguồn).
    """
    where = _web_where(url) if url else None
    ans = answer_query(
        question=question,
        persist_dir=Path(chroma_dir),
        embed_model=embed_model,
        chat_model=chat_model,
        top_k=top_k,
        where=where,
    )
    return {
        "answer": ans,
        "filtered_by_url": url is not None,
        "url": url,
    }


@mcp.tool()
async def atp_run(
    question: str,
    url: Optional[str] = None,
    pdf_dir: Optional[str] = None,
    allowed_domain: Optional[str] = None,
    content_selector: Optional[str] = None,
    chroma_dir: str = str(DEFAULT_CHROMA_DIR),
    embed_model: str = "embeddinggemma",
    chat_model: str = "qwen3:1.7b",
    top_k: int = 4,
    out_dir: str = str(DEFAULT_OUTPUTS_DIR),
) -> dict:
    """
    Pipeline 1 lệnh:
    - pdf_dir: ingest PDF -> query
    - url: scrape -> index -> query (lọc theo url)
    """
    cd = Path(chroma_dir)
    _ensure_dir(cd)

    if (url is None) == (pdf_dir is None):
        return {"ok": False, "error": "Chỉ chọn 1 nguồn: hoặc url hoặc pdf_dir"}

    if pdf_dir:
        dd = Path(pdf_dir)
        pdfs = sorted(dd.glob("*.pdf"))
        if not pdfs:
            return {"ok": False, "error": f"Không thấy PDF trong {pdf_dir}"}

        n = build_vectorstore_from_pdfs(pdfs, cd, embed_model=embed_model)
        ans = answer_query(
            question=question,
            persist_dir=cd,
            embed_model=embed_model,
            chat_model=chat_model,
            top_k=top_k,
        )
        return {"ok": True, "mode": "pdf", "indexed_chunks": n, "answer": ans}

    # url mode
    scrape_result = await atp_web_scrape(
        url=url,
        allowed_domain=allowed_domain,
        content_selector=content_selector,
        headless=True,
        out_dir=out_dir,
    )
    index_result = atp_web_index(
        url=url,
        text_path=str(Path(out_dir) / "page.txt"),
        chroma_dir=chroma_dir,
        embed_model=embed_model,
    )
    ans_obj = atp_rag_query(
        question=question,
        chroma_dir=chroma_dir,
        embed_model=embed_model,
        chat_model=chat_model,
        top_k=top_k,
        url=url,
    )
    return {"ok": True, "mode": "url", "scrape": scrape_result, "index": index_result, "qa": ans_obj}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        default="streamable-http",
        choices=["streamable-http", "stdio"],
        help="Transport cho MCP server",
    )
    args = parser.parse_args()

    # streamable-http: khuyến nghị, dễ test bằng inspector :contentReference[oaicite:4]{index=4}
    # stdio: dùng để tích hợp Claude Desktop/IDE; nhớ KHÔNG print ra stdout :contentReference[oaicite:5]{index=5}
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
