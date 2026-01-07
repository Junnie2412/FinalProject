from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class RetrievedHit:
    page_content: str
    metadata: dict


def load_pdfs(pdf_paths: Iterable[Path]):
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    return docs


def extract_pdfs_text(pdf_paths: Iterable[Path]) -> str:
    """Dùng để dump text kiểm tra loader có đọc được không."""
    docs = load_pdfs(pdf_paths)
    parts = []
    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", "")
        parts.append(f"\n=== SOURCE: {src} | PAGE: {page} ===\n")
        parts.append(d.page_content or "")
    return "\n".join(parts).strip()


def _make_splitter(chunk_size: int, chunk_overlap: int):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_vectorstore_from_pdfs(
    pdf_paths: List[Path],
    persist_dir: Path,
    embed_model: str = "embeddinggemma",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> int:
    raw_docs = load_pdfs(pdf_paths)
    splitter = _make_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(raw_docs)

    emb = OllamaEmbeddings(model=embed_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=str(persist_dir),
    )
    return len(chunks)


def add_textfile_to_vectorstore(
    text_path: Path,
    persist_dir: Path,
    embed_model: str = "embeddinggemma",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    metadata: Optional[dict] = None,
) -> int:
    loader = TextLoader(str(text_path), encoding="utf-8")
    docs = loader.load()
    if metadata:
        for d in docs:
            d.metadata.update(metadata)

    splitter = _make_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)

    emb = OllamaEmbeddings(model=embed_model)
    db = Chroma(persist_directory=str(persist_dir), embedding_function=emb)
    db.add_documents(chunks)
    return len(chunks)


def retrieve_hits(
    question: str,
    persist_dir: Path,
    embed_model: str = "embeddinggemma",
    top_k: int = 4,
    where: Optional[dict] = None,
) -> List[RetrievedHit]:
    """
    where: filter theo metadata.

    Ví dụ lọc đúng 1 URL (Chroma yêu cầu 1 operator):
      where={"$and": [{"source_type": "web"}, {"url": "https://..."}]}
    """
    emb = OllamaEmbeddings(model=embed_model)
    db = Chroma(persist_directory=str(persist_dir), embedding_function=emb)

    # Tương thích nhiều phiên bản: ưu tiên filter=..., fallback where=...
    try:
        docs = db.similarity_search(question, k=top_k, filter=where)
    except TypeError:
        docs = db.similarity_search(question, k=top_k, where=where)

    return [RetrievedHit(page_content=d.page_content, metadata=d.metadata) for d in docs]


def answer_query(
    question: str,
    persist_dir: Path,
    embed_model: str = "embeddinggemma",
    chat_model: str = "qwen3:1.7b",
    top_k: int = 4,
    where: Optional[dict] = None,
) -> str:
    hits = retrieve_hits(
        question=question,
        persist_dir=persist_dir,
        embed_model=embed_model,
        top_k=top_k,
        where=where,
    )
    context = "\n\n---\n\n".join([h.page_content for h in hits])

    llm = OllamaLLM(model=chat_model)
    prompt = f"""Bạn chỉ trả lời dựa trên NGỮ CẢNH. Nếu NGỮ CẢNH không chứa thông tin cần thiết, hãy trả lời đúng một câu: "không đủ thông tin".

NGỮ CẢNH:
{context}

CÂU HỎI: {question}
TRẢ LỜI:"""
    return llm.invoke(prompt)
