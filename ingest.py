"""Step 1 + 2: read PDFs page by page, then cut the text into chunks.

Every chunk remembers WHICH paper and WHICH page it came from -
that is what makes citations possible later.
"""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    source: str   # e.g. "Paper 2.pdf"
    page: int     # 1-based page number


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for one PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pdf(pdf_path: str | Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split one PDF into overlapping chunks, keeping source + page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    name = Path(pdf_path).name
    chunks: list[Chunk] = []
    for page_no, text in extract_pages(pdf_path):
        for piece in splitter.split_text(text):
            chunks.append(Chunk(text=piece, source=name, page=page_no))
    return chunks
