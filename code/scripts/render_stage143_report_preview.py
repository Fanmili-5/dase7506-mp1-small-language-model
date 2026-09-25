"""Render the Stage143 Markdown development draft to a clearly marked PDF preview."""
from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
import textwrap

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (HRFlowable, KeepTogether, LongTable, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer,
                                TableStyle)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "REPORT_STAGE143_DRAFT.md"
DEFAULT_OUTPUT = ROOT / "output/pdf/STAGE143_REPORT_PREVIEW_NOT_FOR_SUBMISSION.pdf"
FONT_DIR = Path(reportlab.__file__).resolve().parent / "fonts"
INLINE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*")


def register_fonts() -> None:
    for name, file in (("Vera", "Vera.ttf"),
                       ("Vera-Bold", "VeraBd.ttf"),
                       ("Vera-Italic", "VeraIt.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold",
                                  italic="Vera-Italic", boldItalic="Vera-Bold")


def safe_text(value: str) -> str:
    return (value.replace("—", "-").replace("–", "-")
            .replace("‑", "-").replace("−", "-"))


def inline_markup(value: str) -> str:
    value = safe_text(value)
    pieces = []
    cursor = 0
    for match in INLINE.finditer(value):
        pieces.append(html.escape(value[cursor:match.start()]))
        label, url, strong, code, italic = match.groups()
        if label is not None:
            label = html.escape(label)
            if url.startswith(("http://", "https://")):
                pieces.append(f'<link href="{html.escape(url, quote=True)}" color="#14628c">{label}</link>')
            else:
                pieces.append(f'<font color="#14628c">{label}</font>')
        elif strong is not None:
            pieces.append(f"<b>{html.escape(strong)}</b>")
        elif code is not None:
            pieces.append(f'<font face="Courier" size="8">{html.escape(code)}</font>')
        else:
            pieces.append(f"<i>{html.escape(italic)}</i>")
        cursor = match.end()
    pieces.append(html.escape(value[cursor:]))
    return "".join(pieces)


def styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("title", fontName="Vera-Bold", fontSize=16,
                                leading=20, textColor=colors.HexColor("#17354f"),
                                alignment=TA_LEFT, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName="Vera-Bold", fontSize=11.3,
                             leading=14, textColor=colors.HexColor("#17354f"),
                             spaceBefore=10, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="Vera", fontSize=8.6,
                               leading=11.9, textColor=colors.HexColor("#182630"),
                               spaceAfter=6),
        "bullet": ParagraphStyle("bullet", fontName="Vera", fontSize=8.4,
                                 leading=11.5, leftIndent=12, firstLineIndent=-8,
                                 textColor=colors.HexColor("#182630"), spaceAfter=4),
        "table_head": ParagraphStyle("table_head", fontName="Vera-Bold",
                                     fontSize=7.7, leading=9.8, textColor=colors.white),
        "table": ParagraphStyle("table", fontName="Vera", fontSize=7.6,
                                leading=9.8, textColor=colors.HexColor("#182630")),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=6.7,
                               leading=8.5, leftIndent=7, rightIndent=7,
                               backColor=colors.HexColor("#f3f6f8"),
                               borderPadding=6, spaceAfter=7),
        "footer": ParagraphStyle("footer", fontName="Vera", fontSize=7,
                                 leading=9, textColor=colors.HexColor("#657888"),
                                 alignment=TA_CENTER),
    }


def table_flow(lines: list[str], sty: dict[str, ParagraphStyle]) -> LongTable:
    raw = [[cell.strip() for cell in line.strip().strip("|").split("|")]
           for line in lines]
    rows = [raw[0]] + raw[2:]
    if any(len(row) != 3 for row in rows):
        raise ValueError("Expected three columns in the report ablation table")
    body = [[Paragraph(inline_markup(cell), sty["table_head"] if i == 0
                       else sty["table"]) for cell in row]
            for i, row in enumerate(rows)]
    table = LongTable(body, colWidths=[202, 113, 190], repeatRows=1,
                      hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244e69")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f3f6f8"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), .4,
         colors.HexColor("#cad6dd")),
    ]))
    return table


def make_story(source: str, sty: dict[str, ParagraphStyle]) -> list:
    lines = source.splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), sty["title"]))
            story.append(HRFlowable(width="100%", thickness=.8,
                                    color=colors.HexColor("#9db5c4"),
                                    spaceAfter=9))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), sty["h2"]))
            i += 1
            continue
        if line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.extend(textwrap.wrap(safe_text(lines[i]), width=108,
                                           break_long_words=False,
                                           replace_whitespace=False) or [""])
                i += 1
            if i == len(lines):
                raise ValueError("Unclosed code fence")
            story.append(Preformatted("\n".join(block), sty["code"]))
            i += 1
            continue
        if line.startswith("| "):
            block = []
            while i < len(lines) and lines[i].startswith("| "):
                block.append(lines[i])
                i += 1
            story.append(table_flow(block, sty))
            story.append(Spacer(1, 7))
            continue
        bullet = line.startswith("- ")
        block = [line[2:] if bullet else line]
        i += 1
        while i < len(lines) and lines[i].strip():
            next_line = lines[i].strip()
            if next_line.startswith(("## ", "# ", "| ", "```", "- ")):
                break
            block.append(next_line)
            i += 1
        content = inline_markup(" ".join(block))
        if bullet:
            content = "&#8226; " + content
        story.append(Paragraph(content, sty["bullet" if bullet else "body"]))
    return story


def on_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd8e1"))
    canvas.line(45, 37, A4[0] - 45, 37)
    canvas.setFont("Vera", 7)
    canvas.setFillColor(colors.HexColor("#657888"))
    canvas.drawString(45, 25, "STAGE143 DEVELOPMENT PREVIEW - NOT FOR SUBMISSION")
    canvas.drawRightString(A4[0] - 45, 25, str(doc.page))
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    register_fonts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=A4,
                            leftMargin=45, rightMargin=45,
                            topMargin=46, bottomMargin=50,
                            title="Stage143 development report preview",
                            author="DASE7506 MP1 working draft")
    doc.build(make_story(SOURCE.read_text(encoding="utf-8"), styles()),
              onFirstPage=on_page, onLaterPages=on_page)
    print(args.output)


if __name__ == "__main__":
    main()
