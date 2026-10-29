"""PDF text extraction and section selection."""

from __future__ import annotations

import pytest

from pdf_text import clean_text, extract_text, select_sections


def minimal_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def test_extract_text_reads_a_simple_pdf():
    text = extract_text(minimal_pdf("Introduction sample"))
    assert "Introduction" in text
    assert "sample" in text


def test_extract_text_rejects_non_pdf():
    with pytest.raises(ValueError):
        extract_text(b"not a pdf")


def test_select_sections_keeps_the_reading_set_and_drops_references():
    text = """
Title

Abstract
We propose a model for translation.

1 Introduction
The field is large enough to need a new method.

2 Related Work
Many papers exist and should not be summarized.

3 Method
We train a transformer on a public corpus.

4. Experimental Results
The model improves accuracy on the benchmark described here.

5 Conclusion
The method helps language tasks.

References
[1] Someone else.
"""
    selected = select_sections(text, 5000)
    assert "We propose a model" in selected
    assert "need a new method" in selected
    assert "train a transformer" in selected
    assert "improves accuracy" in selected
    assert "helps language tasks" in selected
    assert "Many papers exist" not in selected
    assert "Someone else" not in selected


def test_custom_named_sections_after_related_work_are_kept():
    text = """
Abstract
We propose a planner.

1 Introduction
Planning is hard.

2 Related Work
Prior planners exist.
3 ResNet-50 76.1 92.9
We compare against them in detail.
2.1 Learned Planners
Subsection of related work.
2.2 Results of Earlier Systems
Another related subsection.

3 Proposed Framework
Our planner searches with a learned value function.
3.1 Value Network
A subsection of the method.

III. ABLATION STUDY
Removing the value function costs eleven points.

4 Results on Long Horizons
The planner solves longer tasks.

References
[1] A citation.

Appendix
1 Additional Proofs
A long proof that should not be read.
"""
    selected = select_sections(text, 5000)
    assert "learned value function" in selected
    assert "A subsection of the method." in selected
    assert "Subsection of related work." not in selected
    assert "Another related subsection." not in selected
    assert "costs eleven points" in selected
    assert "solves longer tasks" in selected
    assert "Planning is hard." in selected
    assert "Prior planners exist." not in selected
    assert "compare against them" not in selected
    assert "A citation." not in selected
    assert "long proof" not in selected


def test_background_is_kept_and_sentences_with_numbers_are_not_headings():
    text = """
1 Introduction
We study agents.
2 We propose a new benchmark for agents
that continues the introduction.

2 Background and Notation
An agent acts in an environment.
"""
    selected = select_sections(text, 5000)
    assert "continues the introduction" in selected
    assert "acts in an environment" in selected


def test_clean_text_removes_control_characters_and_lone_surrogates():
    assert clean_text("e\x0ecient\x00 model\ud800 \ttab\nline") == "ecient model \ttab\nline"


def test_sentence_starting_with_results_is_not_a_heading():
    text = "results are strong today and tomorrow.\n" + ("middle " * 400) + "\nENDTOKEN"
    selected = select_sections(text, 80)
    assert "ENDTOKEN" in selected


def test_missing_headings_keep_the_start_and_end():
    text = "STARTOKEN " + ("middle " * 500) + " ENDTOKEN"
    selected = select_sections(text, 80)
    assert "STARTOKEN" in selected
    assert "ENDTOKEN" in selected
    assert len(selected) < len(text)
