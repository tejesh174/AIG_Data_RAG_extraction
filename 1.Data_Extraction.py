"""
Downloading AIG 2025 10-K filing from SEC EDGAR.
Source: https://www.sec.gov/Archives/edgar/data/5272/000000527226000023/aig-20251231.htm

"""

import sys, json, urllib.request, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, COMPANY_NAME, FISCAL_YEAR, FILING_HTML_URL

HEADERS = {"User-Agent": "Tejesh Kumar tejesh@gmail.com"}
OUTPUT_FILE = RAW_DIR / f"{COMPANY_NAME}_{FISCAL_YEAR}.txt"

#fetching the url
def fetch_text(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    try:    return raw.decode("utf-8")
    except: return raw.decode("latin-1")


def run():
    print(f"\n{'='*60}")
    print(f" Downloading {COMPANY_NAME} {FISCAL_YEAR} 10-K from SEC EDGAR")
    print(f" Output: {RAW_DIR}")
    print(f"{'='*60}\n")

    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 10_000:
        print(f"  [SKIP] {OUTPUT_FILE.name} ({OUTPUT_FILE.stat().st_size:,} bytes)")
        return str(OUTPUT_FILE)

    # Downloading the filing
    print(f"  Downloading: {FILING_HTML_URL}")
    html = fetch_text(FILING_HTML_URL)
    print(f"  Downloaded {len(html):,} chars")

    # Stripping HTML tags to get the plain text
    text = re.sub(r'<[^>]+>', ' ', html)
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
    text = re.sub(r'\s+', ' ', text).strip()

    # Saving the output file
    OUTPUT_FILE.write_text(text, encoding="utf-8")
    (RAW_DIR / "manifest.json").write_text(json.dumps({"file": str(OUTPUT_FILE), "chars": len(text)}, indent=2))
    print(f"  Saved: {OUTPUT_FILE.name} ({len(text):,} chars)")
    print(f"\nDone.\n")
    return str(OUTPUT_FILE)


if __name__ == "__main__":
    run()