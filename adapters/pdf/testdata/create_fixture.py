"""Build the synthetic two-page parser fixture; no user source documents."""
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.constants import PageLabelStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

BASE = Path(__file__).resolve().parent


def create():
    pixels = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(pixels)
    draw.rectangle((20, 20, 180, 180), outline="black", width=3)
    draw.line((20, 100, 180, 100), fill="black", width=3)
    draw.line((100, 20, 100, 180), fill="black", width=3)
    pixels.save(BASE / "diagram.png")
    memory = BytesIO()
    pdf = canvas.Canvas(memory, pagesize=(600, 800), invariant=1)
    pdf.setTitle("Ragy synthetic retained-source fixture")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(60, 710, "Alpha beta. Gamma.")
    # One merged cell over two columns, followed by two ordinary cells.
    for y in (110, 150, 180):
        pdf.line(60, 800-y, 260, 800-y)
    for x in (60, 260):
        pdf.line(x, 690, x, 620)
    pdf.line(160, 650, 160, 620)
    pdf.drawString(70, 665, "Revenue")
    pdf.drawString(70, 630, "2023")
    pdf.drawString(170, 630, "2024")
    pdf.drawImage(ImageReader(pixels), 100, 400, width=200, height=200)
    pdf.showPage()
    # Raster source has no text layer or promised OCR. Rotation is applied below.
    pdf.drawImage(ImageReader(pixels), 100, 400, width=200, height=200)
    pdf.showPage()
    pdf.save()
    writer = PdfWriter()
    writer.clone_document_from_reader(PdfReader(memory))
    writer.pages[1].rotate(90)
    writer.set_page_label(0, 0, style=PageLabelStyle.LOWERCASE_ROMAN, start=1)
    writer.set_page_label(1, 1, style=PageLabelStyle.DECIMAL, start=1)
    with (BASE / "manual.pdf").open("wb") as output:
        writer.write(output)


if __name__ == "__main__":
    create()
