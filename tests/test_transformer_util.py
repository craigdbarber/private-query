"""Test suite for transformer_util."""

import random
import tempfile
from pathlib import Path

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Flowable, PageBreak, Paragraph, SimpleDocTemplate

from transformer_util import extract_text_from_pdf


def test_extract_text_from_pdf():
    """Test extract_text_from_pdf."""
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".pdf") as tmp_pdf:
        tmp_path = Path(tmp_pdf.name)
        input_page_texts = [
            " ".join(random.choices(["foo", "bar", "baz", "fuzz"], k=100)),
            " ".join(random.choices(["foo", "bar", "baz", "fuzz"], k=100)),
            " ".join(random.choices(["foo", "bar", "baz", "fuzz"], k=100)),
        ]
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

        # verify the cleaned texts are the same
        assert clean_expected == clean_extracted

        # verify page layout order by checking sequential appearance
        # assert extracted_raw.find(input_page_texts[0]) < extracted_raw.find(
        #    input_page_texts[1]
        # )
        # assert extracted_raw.find(input_page_texts[1]) < extracted_raw.find(
        #    input_page_texts[2]
        # )
