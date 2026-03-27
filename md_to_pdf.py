#!/usr/bin/env python3
"""通用 Markdown → PDF 转换器（支持中文，reportlab）
用法: python3 md_to_pdf.py <file.md>
输出: 同目录下 <file.pdf>，完成后自动打开
"""
import sys, os, re, subprocess

# ── 依赖检查 ─────────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
except ImportError:
    print("❌ 需要 reportlab: pip3 install reportlab")
    sys.exit(1)

# ── 中文字体 ──────────────────────────────────────────────────────────────────
FONT_SPECS = [
    ("/Library/Fonts/Arial Unicode.ttf", None),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
]
FONT_NAME = "Chinese"
for fp, idx in FONT_SPECS:
    if os.path.exists(fp):
        try:
            if idx is not None:
                pdfmetrics.registerFont(TTFont(FONT_NAME, fp, subfontIndex=idx))
            else:
                pdfmetrics.registerFont(TTFont(FONT_NAME, fp))
            break
        except Exception:
            continue
else:
    FONT_NAME = "Helvetica"

# ── 颜色 ──────────────────────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#1A1A2E")
C_BLUE   = colors.HexColor("#1E6FD9")
C_LIGHT  = colors.HexColor("#F6F8FA")
C_BORDER = colors.HexColor("#D0D7DE")
C_GRAY   = colors.HexColor("#57606A")
C_WHITE  = colors.white

W, H = A4
MARGIN = 20 * mm

# ── 段落样式 ──────────────────────────────────────────────────────────────────
def S(name, **kw):
    kw.setdefault("fontName", FONT_NAME)
    return ParagraphStyle(name, **kw)

sH1    = S("H1", fontSize=20, textColor=C_BLUE,  spaceAfter=6,  spaceBefore=14, leading=28)
sH2    = S("H2", fontSize=15, textColor=C_DARK,  spaceAfter=5,  spaceBefore=10, leading=22,
           borderPad=0)
sH3    = S("H3", fontSize=12, textColor=C_DARK,  spaceAfter=4,  spaceBefore=8,  leading=18)
sBody  = S("Bd", fontSize=10, textColor=C_DARK,  spaceAfter=4,  leading=16)
sBullet= S("Bl", fontSize=10, textColor=C_DARK,  spaceAfter=3,  leading=16, leftIndent=14)
sQuote = S("Qt", fontSize=10, textColor=C_GRAY,  spaceAfter=4,  leading=16, leftIndent=16,
           borderPad=4, backColor=C_LIGHT)
sCode  = ParagraphStyle("Cd", fontName="Courier", fontSize=8.5,
           textColor=colors.HexColor("#24292F"), backColor=C_LIGHT,
           spaceAfter=3, leading=13, leftIndent=8, rightIndent=8)
sMeta  = S("Mt", fontSize=8,  textColor=C_GRAY,  alignment=TA_CENTER, leading=12)

def safe(text):
    """转义 ReportLab XML 特殊字符，保留 **bold** → <b>"""
    # 先提取 `code` 片段，用占位符保护（避免内部 * 被识别为 italic）
    code_spans = []
    def replace_code(m):
        code_spans.append(m.group(1))
        return f'\x00CODE{len(code_spans)-1}\x00'
    text = re.sub(r'`(.+?)`', replace_code, text)

    text = (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
    # **bold**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # *italic*
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)

    # 还原 code 占位符
    for i, code in enumerate(code_spans):
        escaped = (code.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))
        text = text.replace(f'\x00CODE{i}\x00', f'<font name="Courier">{escaped}</font>')
    return text

# ── 表格解析 ──────────────────────────────────────────────────────────────────
def parse_table(lines):
    rows = []
    for line in lines:
        if re.match(r'^\s*\|[-:| ]+\|\s*$', line):
            continue  # 分隔行
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rows.append(cells)
    if not rows:
        return None

    col_n = max(len(r) for r in rows)
    col_w = (W - 2 * MARGIN) / col_n

    def cell_para(text, is_header=False):
        style = S("tc", fontSize=9, textColor=C_WHITE if is_header else C_DARK,
                  fontName=FONT_NAME, leading=13)
        return Paragraph(safe(text), style)

    data = []
    for i, row in enumerate(rows):
        while len(row) < col_n:
            row.append("")
        data.append([cell_para(c, is_header=(i == 0)) for c in row])

    t = Table(data, colWidths=[col_w] * col_n, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_BLUE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C_WHITE, C_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return t

# ── Markdown 逐行解析 → ReportLab 元素 ────────────────────────────────────────
def md_to_elements(md_text):
    lines = md_text.splitlines()
    elems = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 代码块
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            for cl in code_lines:
                elems.append(Paragraph(cl.replace(" ", "&nbsp;") or "&nbsp;", sCode))
            elems.append(Spacer(1, 3))
            i += 1
            continue

        # 表格（连续 | 行）
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            t = parse_table(table_lines)
            if t:
                elems.append(t)
                elems.append(Spacer(1, 4))
            continue

        # H1
        m = re.match(r'^# (.+)', line)
        if m:
            elems.append(Paragraph(safe(m.group(1)), sH1))
            elems.append(HRFlowable(width="100%", thickness=1.5, color=C_BLUE))
            elems.append(Spacer(1, 2))
            i += 1; continue

        # H2
        m = re.match(r'^## (.+)', line)
        if m:
            elems.append(Paragraph(safe(m.group(1)), sH2))
            elems.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
            elems.append(Spacer(1, 2))
            i += 1; continue

        # H3
        m = re.match(r'^### (.+)', line)
        if m:
            elems.append(Paragraph(safe(m.group(1)), sH3))
            i += 1; continue

        # 分隔线
        if re.match(r'^---+\s*$', line):
            elems.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER))
            elems.append(Spacer(1, 3))
            i += 1; continue

        # 引用块
        if line.strip().startswith("> "):
            text = line.strip()[2:]
            elems.append(Paragraph(safe(text), sQuote))
            i += 1; continue

        # 有序列表
        m = re.match(r'^\d+\. (.+)', line)
        if m:
            elems.append(Paragraph(f"• {safe(m.group(1))}", sBullet))
            i += 1; continue

        # 无序列表
        m = re.match(r'^[-*+] (.+)', line.lstrip())
        if m:
            indent = len(line) - len(line.lstrip())
            li_style = S("li", fontSize=10, textColor=C_DARK, leading=16,
                         leftIndent=14 + indent * 2, fontName=FONT_NAME, spaceAfter=3)
            elems.append(Paragraph(f"• {safe(m.group(1))}", li_style))
            i += 1; continue

        # 空行
        if not line.strip():
            elems.append(Spacer(1, 5))
            i += 1; continue

        # 普通段落
        elems.append(Paragraph(safe(line), sBody))
        i += 1

    return elems

# ── 页眉页脚 ──────────────────────────────────────────────────────────────────
def on_page(canvas, doc, title=""):
    canvas.saveState()
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, H - 8 * mm, W, 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont(FONT_NAME, 7.5)
    canvas.drawString(MARGIN, H - 5 * mm, title)
    canvas.drawRightString(W - MARGIN, H - 5 * mm, f"第 {doc.page} 页")
    canvas.restoreState()

# ── 主函数 ────────────────────────────────────────────────────────────────────
def convert(md_path):
    if not os.path.exists(md_path):
        print(f"❌ 文件不存在: {md_path}")
        sys.exit(1)

    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    # 输出路径：同目录，同名 .pdf
    base = os.path.splitext(md_path)[0]
    out_path = base + ".pdf"

    # 从第一个 H1 取标题
    m = re.search(r'^# (.+)', md_text, re.MULTILINE)
    title = m.group(1) if m else os.path.basename(base)

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16 * mm, bottomMargin=12 * mm,
        title=title,
    )

    story = md_to_elements(md_text)

    doc.build(story, onFirstPage=lambda c, d: on_page(c, d, title),
                     onLaterPages=lambda c, d: on_page(c, d, title))

    print(f"✅ PDF 已生成: {out_path}")
    subprocess.run(["open", out_path])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 md_to_pdf.py <file.md>")
        sys.exit(1)
    for path in sys.argv[1:]:
        convert(path)
