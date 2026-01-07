from __future__ import annotations

from pathlib import Path
from typing import Optional

from atp.rag.rag_core import add_textfile_to_vectorstore


def index_web_text(
    text_path: Path,
    chroma_dir: Path,
    url: str,
    embed_model: str = "embeddinggemma",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    extra_metadata: Optional[dict] = None,
) -> int:
    metadata = {"source_type": "web", "url": url}
    if extra_metadata:
        metadata.update(extra_metadata)

    return add_textfile_to_vectorstore(
        text_path=text_path,
        persist_dir=chroma_dir,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata=metadata,
    )
