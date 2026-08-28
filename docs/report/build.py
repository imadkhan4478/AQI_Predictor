"""Render report.html to PDF. Two passes: the first finds which page each section
lands on, the second injects a table of contents with those numbers."""
import re, pathlib, pdfplumber
from playwright.sync_api import sync_playwright

HTML = pathlib.Path("report.html")
OUT = pathlib.Path("AQI_Predictor_Technical_Report.pdf")
TMP = pathlib.Path("_pass1.pdf")

def render(html_path, out_path):
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.goto(f"file://{html_path.resolve()}", wait_until="load")
        page.emulate_media(media="print")
        page.pdf(path=str(out_path), format="A4", print_background=True,
                 margin={"top": "20mm", "bottom": "18mm", "left": "18mm", "right": "18mm"},
                 display_header_footer=False)
        b.close()

source = HTML.read_text(encoding="utf-8")

# Sections and subsections, in document order, from the headings themselves.
# Use the heading's own text as both the label and the search key, so the two
# cannot disagree about whether a number is followed by a dot.
headings = []
for level, inner in re.findall(r'<h([23])[^>]*>(.*?)</h\1>', source, re.S):
    plain = re.sub(r'<[^>]+>', '', inner).strip()
    m = re.match(r'^(\d+(?:\.\d+)?)\.?\s+(.*)$', plain)
    if m:
        headings.append((level, m.group(1), m.group(2), plain))
figures = re.findall(r'<figcaption><b>(Figure\s+\d+)\s*&mdash;\s*([^<.]+)', source)
if not figures:
    figures = re.findall(r'<figcaption><b>(Figure\s+\d+)\s*—\s*([^<.]+)', source)

render(HTML, TMP)

pages = {}
with pdfplumber.open(TMP) as pdf:
    text = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(pdf.pages)]
def find(needle):
    probe = " ".join(needle.split())[:40]
    for pageno, body in text:
        if probe in " ".join(body.split()):
            return pageno
    return ""

for level, num, title, plain in headings:
    pages[plain] = find(plain)
figure_pages = {f"{label} {title}": find(f"{label} — {title}") or find(label) for label, title in figures}

rows = []
for level, num, title, plain in headings:
    cls = "grp" if level == "2" else "sub"
    rows.append(
        f'<div class="{cls}"><span class="n">{num}</span><span class="t">{title}</span>'
        f'<span class="dots"></span><span class="p">{pages.get(plain, "")}</span></div>'
    )
toc = "\n".join(rows)

lof = "\n".join(
    f'<div><span class="n">{label.split()[1]}</span><span class="t">{title.strip()}</span>'
    f'<span class="dots"></span><span class="p">{figure_pages.get(f"{label} {title}", "")}</span></div>'
    for label, title in figures
)

final = source.replace('<div class="toc" id="toc"></div>', f'<div class="toc">{toc}</div>')
final = final.replace('<div class="toc" id="lof"></div>', f'<div class="toc">{lof}</div>')
tmp_html = pathlib.Path("_final.html")
tmp_html.write_text(final, encoding="utf-8")
render(tmp_html, OUT)
TMP.unlink(missing_ok=True)

with pdfplumber.open(OUT) as pdf:
    print(f"{OUT.name}: {len(pdf.pages)} pages")
ok_h = sum(1 for _, _, _, plain in headings if pages.get(plain))
ok_f = sum(1 for label, title in figures if figure_pages.get(f"{label} {title}"))
print(f"contents: {ok_h}/{len(headings)} resolved  |  figures: {ok_f}/{len(figures)} resolved")
