from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import typer
from rich import print

from atp.rag.rag_core import (
    add_textfile_to_vectorstore,
    answer_query,
    build_vectorstore_from_pdfs,
    extract_pdfs_text,
    retrieve_hits,
)
from atp.web.index import index_web_text
from atp.web.scrape import scrape_url
from atp.web.search import search_urls

app = typer.Typer(no_args_is_help=True)

DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_DOCS_DIR = Path("docs")


def _write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@app.command()
def web_search(
    query: str = typer.Argument(..., help="Từ khoá tìm URL"),
    limit: int = typer.Option(5, help="Số URL trả về"),
    allowed_domain: Optional[List[str]] = typer.Option(
        None, help="Allowlist domain (lặp nhiều lần)"
    ),
):
    urls = search_urls(query=query, limit=limit, allowed_domains=allowed_domain)
    for i, u in enumerate(urls, 1):
        print(f"{i}. {u}")


@app.command()
def web_scrape(
    url: str = typer.Argument(..., help="URL cần lấy"),
    out_dir: Path = typer.Option(DEFAULT_OUTPUTS_DIR, help="Thư mục output"),
    allowed_domain: Optional[List[str]] = typer.Option(
        None, help="Allowlist domain (lặp nhiều lần)"
    ),
    headless: bool = typer.Option(
        True, help="Chạy headless (dùng --no-headless để mở browser)"
    ),
    content_selector: Optional[str] = typer.Option(
        None, help='CSS selector lấy nội dung chính, ví dụ: "article"'
    ),
    timeout_ms: int = typer.Option(30000, help="Timeout (ms)"),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    async def _run():
        r = await scrape_url(
            url,
            allowed_domains=allowed_domain,
            headless=headless,
            timeout_ms=timeout_ms,
            content_selector=content_selector,
        )
        _write_text(out_dir / "page.html", r.html)
        _write_text(out_dir / "page.txt", r.text)
        print(f"[green]OK[/green] Saved: {out_dir/'page.html'} and {out_dir/'page.txt'}")

    asyncio.run(_run())


@app.command()
def web_index(
    text_path: Path = typer.Option(
        Path("outputs/page.txt"), help="Đường dẫn file text đã scrape"
    ),
    url: str = typer.Option(..., help="URL nguồn để gắn metadata"),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Chroma persist dir"),
    embed_model: str = typer.Option("embeddinggemma", help="Ollama embedding model"),
):
    if not text_path.exists():
        raise typer.BadParameter(f"Không thấy file: {text_path}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    n = index_web_text(
        text_path=text_path,
        chroma_dir=chroma_dir,
        url=url,
        embed_model=embed_model,
    )
    print(f"[green]OK[/green] Added {n} chunks from web text into {chroma_dir}")


@app.command()
def rag_ingest(
    docs_dir: Path = typer.Option(DEFAULT_DOCS_DIR, help="Thư mục chứa PDF"),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Chroma persist dir"),
    embed_model: str = typer.Option("embeddinggemma", help="Ollama embedding model"),
    dump_text: bool = typer.Option(False, help="Xuất text PDF ra file để kiểm tra"),
    out_dir: Path = typer.Option(DEFAULT_OUTPUTS_DIR, help="Thư mục output (khi dump_text)"),
):
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise typer.BadParameter(f"Không thấy PDF trong {docs_dir}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    n = build_vectorstore_from_pdfs(pdfs, chroma_dir, embed_model=embed_model)
    print(f"[green]OK[/green] Indexed {n} chunks into {chroma_dir}")

    if dump_text:
        out_dir.mkdir(parents=True, exist_ok=True)
        txt = extract_pdfs_text(pdfs)
        _write_text(out_dir / "pdf_extracted.txt", txt)
        print(f"[green]OK[/green] Dumped extracted PDF text to {out_dir/'pdf_extracted.txt'}")


@app.command()
def rag_query(
    question: str = typer.Argument(..., help="Câu hỏi"),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Chroma persist dir"),
    embed_model: str = typer.Option("embeddinggemma", help="Ollama embedding model"),
    chat_model: str = typer.Option("qwen3:1.7b", help="Ollama chat model"),
    top_k: int = typer.Option(4, help="Số chunk truy hồi"),
    out_dir: Path = typer.Option(DEFAULT_OUTPUTS_DIR, help="Thư mục output debug"),
    save_debug: bool = typer.Option(True, help="Lưu context/answer/hits để debug"),
):
    ans = answer_query(
        question=question,
        persist_dir=chroma_dir,
        embed_model=embed_model,
        chat_model=chat_model,
        top_k=top_k,
    )
    print(ans)

    if save_debug:
        hits = retrieve_hits(
            question=question,
            persist_dir=chroma_dir,
            embed_model=embed_model,
            top_k=top_k,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_text(out_dir / "last_question.txt", question)
        _write_text(out_dir / "last_answer.txt", ans)
        _write_text(out_dir / "last_context.txt", "\n\n---\n\n".join(h.page_content for h in hits))
        _write_json(
            out_dir / "last_hits.json",
            [{"metadata": h.metadata, "preview": h.page_content[:400]} for h in hits],
        )
        print(f"[green]OK[/green] Saved debug to {out_dir}")


@app.command()
def run(
    question: str = typer.Argument(..., help="Câu hỏi"),
    pdf_dir: Optional[Path] = typer.Option(None, help="Nếu dùng PDF: thư mục chứa PDF"),
    url: Optional[str] = typer.Option(None, help="Nếu dùng Web: URL"),
    allowed_domain: Optional[List[str]] = typer.Option(None, help="Allowlist domain (web)"),
    content_selector: Optional[str] = typer.Option(
        None, help='Selector nội dung chính, ví dụ "article"'
    ),
    chroma_dir: Path = typer.Option(DEFAULT_CHROMA_DIR, help="Chroma persist dir"),
    embed_model: str = typer.Option("embeddinggemma", help="Ollama embedding model"),
    chat_model: str = typer.Option("qwen3:1.7b", help="Ollama chat model"),
    out_dir: Path = typer.Option(DEFAULT_OUTPUTS_DIR, help="Thư mục output"),
    headless: bool = typer.Option(True, help="Web headless (dùng --no-headless để mở browser)"),
    top_k: int = typer.Option(4, help="Số chunk truy hồi"),
):
    """
    Pipeline 1 lệnh:
    - Nếu pdf_dir: ingest PDF -> query
    - Nếu url: scrape -> save outputs/page.* -> index outputs/page.txt -> query (lọc đúng url)
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (pdf_dir is None) == (url is None):
        raise typer.BadParameter("Chọn đúng 1 nguồn: hoặc --pdf-dir hoặc --url")

    # --- PDF mode ---
    if pdf_dir is not None:
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise typer.BadParameter(f"Không thấy PDF trong {pdf_dir}")

        n = build_vectorstore_from_pdfs(pdfs, chroma_dir, embed_model=embed_model)
        print(f"[green]OK[/green] Indexed {n} chunks from PDFs into {chroma_dir}")

        ans = answer_query(
            question=question,
            persist_dir=chroma_dir,
            embed_model=embed_model,
            chat_model=chat_model,
            top_k=top_k,
        )
        print(ans)
        _write_text(out_dir / "last_question.txt", question)
        _write_text(out_dir / "last_answer.txt", ans)
        return

    # --- URL mode ---
    async def _scrape():
        r = await scrape_url(
            url,
            allowed_domains=allowed_domain,
            headless=headless,
            content_selector=content_selector,
        )
        _write_text(out_dir / "page.html", r.html)
        _write_text(out_dir / "page.txt", r.text)

    asyncio.run(_scrape())

    added = add_textfile_to_vectorstore(
        text_path=out_dir / "page.txt",
        persist_dir=chroma_dir,
        embed_model=embed_model,
        metadata={"source_type": "web", "url": url},
    )
    print(f"[green]OK[/green] Added {added} chunks from web into {chroma_dir}")

    # QUAN TRỌNG: Chroma where cần 1 operator -> dùng $and
    where = {"$and": [{"source_type": "web"}, {"url": url}]}

    ans = answer_query(
        question=question,
        persist_dir=chroma_dir,
        embed_model=embed_model,
        chat_model=chat_model,
        top_k=top_k,
        where=where,
    )
    print(ans)
    _write_text(out_dir / "last_question.txt", question)
    _write_text(out_dir / "last_answer.txt", ans)
