"""Build kosychev_midterm_report.pdf from report.md (Markdown -> HTML -> PDF via headless Chrome).

Run:  python report/build_report.py
"""
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT_PDF = HERE / "kosychev_midterm_report.pdf"

CSS = """
@page { size: A4; }
body { font-family: 'Times New Roman', Times, serif; font-size: 9.4pt; line-height: 1.17; color: #111; }
h1 { font-size: 15pt; text-align: center; margin: 0 0 5px; line-height: 1.18; }
.meta { text-align: center; font-style: italic; font-size: 9pt; margin-bottom: 8px; }
h2 { font-size: 11pt; margin: 9px 0 3px; color: #1f3b63; border-bottom: 1px solid #c9d3e0; break-after: avoid; }
h3 { font-size: 9.8pt; margin: 6px 0 2px; }
p { margin: 2px 0 4px; text-align: justify; }
ul, ol { margin: 2px 0 4px; padding-left: 16px; }
li { margin: 1px 0; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.2pt; background: #f2f4f7; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 3px 0 5px; }
th, td { border: 1px solid #8fa3bf; padding: 1px 4px; text-align: left; vertical-align: top; }
th { background: #dde6f2; }
table.ex td { font-size: 7.4pt; }
.fig { text-align: center; margin: 4px 0; page-break-inside: avoid; }
.fig img { width: 48%; margin: 0 0.5%; border: 1px solid #ccc; }
.fig.full img { width: 80%; }
.fig.shot img { width: 70%; }
.fig.kaggle img { vertical-align: top; margin: 0 0.5%; }
.fig.kaggle img:first-child { width: 39.5%; }
.fig.kaggle img:last-child { width: 57.5%; }
.fig.hash img { width: 48%; margin: 0 0.5%; vertical-align: top; }
.fig p { text-align: center; font-size: 8.4pt; margin-top: 2px; }
.todo { background: #fff3a8; font-style: italic; }
"""


def main():
    md_text = (HERE / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "md_in_html"])
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Midterm report — Aleksei Kosychev</title><style>{CSS}</style></head>
<body>{body}</body></html>"""
    html_path = HERE / "report.html"
    html_path.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.pdf(path=str(OUT_PDF), format="A4", print_background=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8px;width:100%;text-align:center">'
                                 '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                 margin={"top": "14mm", "bottom": "14mm", "left": "15mm", "right": "15mm"})
        browser.close()
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
