"""Contains utility functions for text transformation."""

from pathlib import Path

import docx
from pypdf import PdfReader


def extract_text_from_docx(docx_path: str | Path) -> str:
    """Read the specified docx file and extract all readable text.

    Args:
        docx_path: The path of the docx file to be read.
    Returns: The extracted readable text.

    Raises:
        FileNotFoundError: If the specified docx path doesn't exist.

    """
    file = Path(docx_path)
    if not file.exists():
        raise FileNotFoundError(f"The docx path: {docx_path} does not exists.")

    doc = docx.Document(str(file))
    extracted_text: list[str] = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            extracted_text.append(paragraph.text)

    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                extracted_text.append("\t".join(row_text))

    return "\n\n".join(extracted_text)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Read the specified pdf file and extract all readable text.

    Args:
        pdf_path: The path of the pdf file to be read.
    Returns: The extracted readable text.

    Raises:
        FileNotFoundError: If the specified pdf path doesn't exist.

    """
    file = Path(pdf_path)
    if not file.exists():
        raise FileNotFoundError(f"The pdf path: {pdf_path} does not exist.")
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
