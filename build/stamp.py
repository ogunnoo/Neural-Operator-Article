"""Add a footer (page numbers + short title) and PDF metadata to the rendered PDF."""
import io
import sys
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color

SRC, DST = sys.argv[1], sys.argv[2]
FOOT_L = "Building a Simple Neural Operator Experiment"
FOOT_R = "DeepONet · lid-driven cavity · Re = 100"

reader = PdfReader(SRC)
n = len(reader.pages)
writer = PdfWriter()
grey = Color(0.42, 0.46, 0.51)
rule = Color(0.85, 0.87, 0.89)

for i, page in enumerate(reader.pages, start=1):
    w = float(page.mediabox.width)
    h = float(page.mediabox.height)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    y = 30                                  # baseline of the footer text
    c.setStrokeColor(rule)
    c.setLineWidth(0.5)
    c.line(51, y + 12, w - 51, y + 12)      # hairline above the footer

    c.setFillColor(grey)
    c.setFont("Helvetica", 7.2)
    c.drawString(51, y, FOOT_L)
    c.drawRightString(w - 51, y, FOOT_R)
    c.setFont("Helvetica-Bold", 7.6)
    c.drawCentredString(w / 2.0, y, f"{i} / {n}")
    c.save()

    buf.seek(0)
    page.merge_page(PdfReader(buf).pages[0])
    writer.add_page(page)

writer.add_metadata({
    "/Title": "Building a Simple Neural Operator Experiment",
    "/Subject": ("Learning the lid-driven-cavity solution operator with a "
                 "DeepONet: g(x) -> [u(x,y), v(x,y)] at fixed Re = 100"),
    "/Keywords": ("neural operator, DeepONet, PINN, lid-driven cavity, "
                  "Navier-Stokes, JAX, operator learning"),
    "/Creator": "pandoc + Chrome headless + reportlab",
})
with open(DST, "wb") as f:
    writer.write(f)
print(f"stamped {n} pages -> {DST}")
