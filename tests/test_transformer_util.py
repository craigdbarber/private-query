"""Test suite for transformer_util."""

import random
import stat
import tempfile
from pathlib import Path

import pytest
from docx import Document
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate

from transformer_util import (
    extract_text_from_docx,
    extract_text_from_file,
    extract_text_from_pdf,
    extract_text_from_txt,
)


def _generate_random_texts(k: int = 100, num_pages: int = 10) -> list[str]:
    page = 0
    pages: list[str] = []
    while page < num_pages:
        pages.append(
            " ".join(random.choices(["foo", "bar", "baz", "fuzz"], k=k)),
        )
        page += 1
    return pages


def test_extract_text_from_file():
    """Test correctly extracts text."""
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as tmp_txt:
        input_page_texts = _generate_random_texts()
        for page in input_page_texts:
            tmp_txt.write(f"{page}\n")
        tmp_txt.flush()

        tmp_path = Path(tmp_txt.name).resolve()
        extracted_raw = extract_text_from_file(tmp_path)
        clean_extracted = " ".join(extracted_raw.split())
        clean_expected = " ".join(" ".join(input_page_texts).split())
        assert clean_extracted == clean_expected


def test_extract_text_from_file_raises_fnf_error():
    """Test raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_file("non existant file")


def test_extract_text_from_file_raises_perm_error():
    """Test raises PermissionError."""
    with (
        pytest.raises(PermissionError),
        tempfile.NamedTemporaryFile() as tmp,
    ):
        tmp_path = Path(tmp.name).resolve()
        tmp_path.touch()
        current_mode = tmp_path.stat().st_mode
        read_bits = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        new_mode = current_mode & ~read_bits
        tmp_path.chmod(mode=new_mode)
        extract_text_from_file(tmp_path)


def test_extract_text_from_file_raises_value_error_unsupported_type():
    """Test raises ValueError for unsupported file type."""
    with (
        pytest.raises(ValueError, match=r"Unsupported file type:.*"),
        tempfile.NamedTemporaryFile(mode="r", suffix=".foobaz") as tmp,
    ):
        tmp_path = Path(tmp.name).resolve()
        tmp_path.touch()
        extract_text_from_file(tmp_path)


def test_extract_text_from_file_raises_value_error_not_file():
    """Test raises ValueError for not a file."""
    with (
        pytest.raises(ValueError, match=r"File path is not file:.*"),
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        tmp_path = Path(tmp_dir).resolve()
        extract_text_from_file(tmp_path)


def test_extract_text_from_txt():
    """Test correctly extracts text."""
    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt") as tmp_txt:
        input_page_texts = _generate_random_texts()
        for page in input_page_texts:
            tmp_txt.write(f"{page}\n")
        tmp_txt.flush()

        tmp_path = Path(tmp_txt.name).resolve()
        extracted_raw = extract_text_from_txt(tmp_path)
        clean_extracted = " ".join(extracted_raw.split())
        clean_expected = " ".join(" ".join(input_page_texts).split())
        assert clean_extracted == clean_expected


def test_extract_text_from_docx():
    """Test correctly extracts text."""
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".docx") as tmp_docx:
        doc = Document()
        input_page_texts = _generate_random_texts()
        for page in input_page_texts:
            doc.add_paragraph(page)
        tmp_path = Path(tmp_docx.name).resolve()
        doc.save(str(tmp_path))
        tmp_docx.flush()

        extracted_raw = extract_text_from_docx(tmp_path)
        clean_extracted = " ".join(extracted_raw.split())
        clean_expected = " ".join(" ".join(input_page_texts).split())
        assert clean_extracted == clean_expected


def test_extract_text_from_pdf():
    """Test correctly extracts text."""
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".pdf") as tmp_pdf:
        tmp_path = Path(tmp_pdf.name)
        input_page_texts = _generate_random_texts()
        doc = SimpleDocTemplate(str(tmp_path))
        styles = getSampleStyleSheet()
        story: list[Flowable] = []
        for i, text in enumerate(input_page_texts):
            story.append(Paragraph(text, styles["Normal"]))
            if i < len(input_page_texts) - 1:
                story.append(PageBreak())
        doc.build(story)
        tmp_pdf.flush()

        # pypdf often injects '\n' or '\x0c' between pages
        extracted_raw = extract_text_from_pdf(tmp_path)
        clean_extracted = " ".join(extracted_raw.split())
        clean_expected = " ".join(" ".join(input_page_texts).split())

        assert clean_expected == clean_extracted


if __name__ == "__main__":
    pytest.main()
