"""Test suite for transformer_util."""

import random
import stat
from collections.abc import Callable
from pathlib import Path

import pytest
from docx import Document
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate

from transformer_util import (
    extract_text_from_file,
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


def test_extract_text_from_file_raises_fnf_error():
    """Test raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_file("non-existent file")


def test_extract_text_from_file_raises_perm_error(tmp_path: Path):
    """Test raises PermissionError."""
    with pytest.raises(PermissionError):
        tmp_file = (tmp_path / "tmp_file").resolve()
        tmp_file.touch()
        current_mode = tmp_file.stat().st_mode
        read_bits = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        new_mode = current_mode & ~read_bits
        tmp_file.chmod(mode=new_mode)
        extract_text_from_file(tmp_file)


def test_extract_text_from_file_raises_value_error_not_file(tmp_path: Path):
    """Test raises ValueError for not a file."""
    with pytest.raises(ValueError, match=r"File path is not file:.*"):
        tmp_dir = (tmp_path / "tmp_dir").resolve()
        tmp_dir.mkdir()
        extract_text_from_file(tmp_dir)


def test_extract_text_from_file_empty_file(tmp_path: Path):
    """Test successfully handles empty file."""
    tmp_file = (tmp_path / "tmp.txt").resolve()
    tmp_file.write_text("")
    extracted_text = extract_text_from_file(tmp_file)
    assert extracted_text is not None
    assert extracted_text == ""


def _generate_txt_file(path: Path, texts: list[str]):
    path.write_text("\n".join(page for page in texts))


def _generate_docx_file(path: Path, texts: list[str]):
    doc = Document()
    for page in texts:
        doc.add_paragraph(page)
    doc.save(str(path))


def _generate_pdf_file(path: Path, texts: list[str]):
    doc = SimpleDocTemplate(str(path))
    styles = getSampleStyleSheet()
    story: list[Flowable] = []
    for i, text in enumerate(texts):
        story.append(Paragraph(text, styles["Normal"]))
        if i < len(texts) - 1:
            story.append(PageBreak())
    doc.build(story)


@pytest.mark.parametrize(
    "extension, generate_func",
    [
        (".txt", _generate_txt_file),
        (".docx", _generate_docx_file),
        (".pdf", _generate_pdf_file),
    ],
)
def test_extract_text_from_file_all_formats(
    tmp_path: Path, extension: str, generate_func: Callable[[Path, list[str]], None]
):
    """Test extract_text_from_file correctly handles all supported formats."""
    # test populated file
    texts = _generate_random_texts()
    tmp_file = (tmp_path / f"tmp_{extension}").resolve()
    generate_func(tmp_file, texts)
    _clean_and_test_expected_text(extract_text_from_file(tmp_file), texts)

    # test empty file
    tmp_empty_file = (tmp_path / f"tmp_empty{extension}").resolve()
    empty_texts: list[str] = []
    generate_func(tmp_empty_file, empty_texts)
    _clean_and_test_expected_text(extract_text_from_file(tmp_empty_file), empty_texts)


def _clean_and_test_expected_text(extracted_raw: str, texts: list[str]):
    clean_extracted = " ".join(extracted_raw.split())
    clean_expected = " ".join(" ".join(texts).split())
    assert clean_expected == clean_extracted
