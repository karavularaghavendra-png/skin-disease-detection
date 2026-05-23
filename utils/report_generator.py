"""
PDF Report Generator for Skin Disease Detection.

Generates a professional clinical-style PDF report containing:
- Uploaded image + Grad-CAM heatmap (if available)
- Prediction result with confidence and severity
- Symptoms and recommendations
- Medical disclaimer

Usage:
    from utils.report_generator import generate_pdf_report
    pdf_bytes = generate_pdf_report(
        image_path="tmp/skin.jpg",
        disease_name="Eczema",
        confidence=87.3,
        severity_label="Moderate",
        symptoms=["Itchy skin", ...],
        recommendations=["Moisturise daily", ...],
        heatmap_path="tmp/gradcam.jpg",   # optional
        tta_agreement=92.5,               # optional
        entropy=0.312,                    # optional
    )
"""

from __future__ import annotations

import os
from datetime import datetime

from fpdf import FPDF


def _sanitize_text(text: str) -> str:
    """Replace Unicode characters unsupported by PDF core fonts with ASCII equivalents.

    The default Helvetica font in fpdf2 only covers the latin-1 range.
    Characters like em-dash, en-dash, curly quotes, and bullets must be
    replaced to prevent rendering errors.
    """
    replacements = {
        "\u2014": "-",    # em-dash —
        "\u2013": "-",    # en-dash –
        "\u2018": "'",    # left single quote '
        "\u2019": "'",    # right single quote '
        "\u201c": '"',    # left double quote "
        "\u201d": '"',    # right double quote "
        "\u2022": "-",    # bullet •
        "\u2026": "...",  # ellipsis …
        "\u00a0": " ",    # non-breaking space
        "\u2192": "->",   # right arrow →
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


class _SkinReportPDF(FPDF):
    """Custom PDF class with header and footer."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(41, 128, 185)  # Steel blue
        self.cell(
            0, 10,
            "Skin Disease Detection Using Deep Learning - Report",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        self.set_draw_color(41, 128, 185)
        self.line(10, 22, 200, 22)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_pdf_report(
    image_path: str,
    disease_name: str,
    confidence: float,
    severity_label: str,
    symptoms: list[str],
    recommendations: list[str],
    heatmap_path: str | None = None,
    tta_agreement: float | None = None,
    entropy: float | None = None,
    latency_ms: float | None = None,
    medications: list[dict] | None = None,
) -> bytes:
    """Generate a PDF report and return it as bytes.

    Args:
        image_path:      Path to the uploaded skin image.
        disease_name:    Predicted disease display name.
        confidence:      Confidence percentage (0-100).
        severity_label:  Severity string (e.g. "Moderate — Monitor Closely").
        symptoms:        List of symptom strings.
        recommendations: List of recommendation strings.
        heatmap_path:    Optional path to Grad-CAM heatmap overlay image.
        tta_agreement:   Optional TTA agreement percentage.
        entropy:         Optional prediction entropy.
        latency_ms:      Optional inference latency in ms.
        medications:     Optional list of {"name": ..., "use": ...} dicts.

    Returns:
        PDF file contents as bytes.
    """
    pdf = _SkinReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Report metadata ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    now = datetime.now().strftime("%B %d, %Y  %I:%M %p")
    pdf.cell(
        0, 6, f"Generated on: {now}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.cell(
        0, 6,
        "Model: MobileNetV2 Transfer Learning + 8-Pass TTA",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(4)

    # ── Images (uploaded + heatmap side by side) ─────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Uploaded Image & Model Attention Heatmap", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    img_w = 80  # width per image in mm
    if os.path.exists(image_path):
        pdf.image(image_path, x=15, w=img_w)

    if heatmap_path and os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=110, y=pdf.get_y() - img_w, w=img_w)

    pdf.ln(5)

    # ── Prediction Result ────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 10, "Prediction Result", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)

    _add_field(pdf, "Detected Condition", _sanitize_text(disease_name))
    _add_field(pdf, "Confidence", f"{confidence:.1f}%")
    _add_field(pdf, "Severity", _sanitize_text(severity_label))

    if tta_agreement is not None:
        _add_field(pdf, "TTA Agreement", f"{tta_agreement:.1f}%")
    if entropy is not None:
        _add_field(pdf, "Prediction Entropy", f"{entropy:.4f}")
    if latency_ms is not None:
        _add_field(pdf, "Inference Latency", f"{latency_ms:.0f} ms")

    pdf.ln(4)

    # ── Symptoms ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 8, "Common Symptoms", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for s in symptoms:
        pdf.cell(5)
        pdf.cell(0, 6, f"  - {_sanitize_text(s)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Recommendations ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 8, "Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for r in recommendations:
        pdf.cell(5)
        pdf.cell(0, 6, f"  - {_sanitize_text(r)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Medications (if any) ─────────────────────────────────────────────
    if medications:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 8, "Suggested OTC Medications", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        for med in medications:
            pdf.cell(5)
            med_text = (
                f"  - {_sanitize_text(med['name'])}"
                f" -- {_sanitize_text(med['use'])}"
            )
            pdf.cell(
                0, 6, med_text,
                new_x="LMARGIN", new_y="NEXT",
            )
        pdf.ln(3)

    # ── Disclaimer ───────────────────────────────────────────────────────
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(200, 50, 50)
    pdf.cell(0, 8, "MEDICAL DISCLAIMER", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, (
        "This report is generated by an AI-assisted screening tool for educational "
        "and informational purposes ONLY. It is NOT a substitute for professional "
        "medical advice, diagnosis, or treatment. Always consult a qualified "
        "dermatologist before taking any medication or starting any treatment. "
        "Do not disregard professional medical advice based on this report."
    ))

    return pdf.output()


def _add_field(pdf: FPDF, label: str, value: str):
    """Add a bold label + normal value line."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(55, 7, f"{label}:")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")
