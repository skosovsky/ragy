"""Verify actual external parser output independently of normalized contract fixtures."""
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

path = Path(__file__).resolve().parent / "manual.pdf"
reader = PdfReader(path)
assert reader.page_labels == ["i", "1"]
with pdfplumber.open(path) as document:
    first, second = document.pages
    assert first.extract_text() == "Alpha beta. Gamma.\nRevenue\n2023 2024"
    assert first.find_tables()[0].extract() == [["Revenue", None], ["2023", "2024"]]
    assert first.find_tables()[0].cells == [
        (60., 110., 260., 150.), (60., 150., 160., 180.), (160., 150., 260., 180.)
    ]
    assert (first.images[0]["x0"], first.images[0]["top"],
            first.images[0]["x1"], first.images[0]["bottom"]) == (100., 200., 300., 400.)
    assert second.rotation == 90 and second.extract_text() == ""
    assert (second.images[0]["x0"], second.images[0]["top"],
            second.images[0]["x1"], second.images[0]["bottom"]) == (400., 100., 600., 300.)
print("synthetic PDF text/table/image/rotation checks passed")
