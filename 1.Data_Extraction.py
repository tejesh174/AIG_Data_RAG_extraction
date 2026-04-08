"""
Downloading AIG 10-K filings from SEC EDGAR.

"""

import sys, json, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, COMPANY_NAME, YEARS

AIG_CIK     = "0000005272"
AIG_CIK_INT = 5272          
EDGAR_DATA  = "https://data.sec.gov"
EDGAR_WWW   = "https://www.sec.gov"
HEADERS = {"User-Agent": "Tejesh Kumar tejesh@gmail.com"}


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def fetch_text(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    try:    return raw.decode("utf-8")
    except: return raw.decode("latin-1")

#retreiving all 10k filings
def all_10k_filings():
    results = []

    def extract(rec):
        forms = rec.get("form", [])
        dates = rec.get("filingDate", [])
        accs  = rec.get("accessionNumber", [])
        pdocs = rec.get("primaryDocument", [rec.get("primaryDocument","")]*len(forms))
        for f, d, a, doc in zip(forms, dates, accs, pdocs):
            if f.upper() in ("10-K", "10-K/A"):
                results.append({
                    "date":            d,
                    "accession_fmt":   a,
                    "accession":       a.replace("-", ""),
                    "primary_doc":     doc,
                })

    subs = fetch_json(f"{EDGAR_DATA}/submissions/CIK{AIG_CIK}.json")
    print(f"  Entity: {subs.get('name')}  CIK={subs.get('cik')}\n")
    extract(subs.get("filings", {}).get("recent", {}))

    for page in subs.get("filings", {}).get("files", []):
        page_data = fetch_json(f"{EDGAR_DATA}/submissions/{page['name']}")
        extract(page_data)
        time.sleep(0.2)

    print(f"  Total 10-K filings found: {len(results)}")
    return results


#finding filings based on fiscal year
def find_filing_for_year(all_filings, fiscal_year):
    #10-K for fiscal year N is filed in year N+1
    for f in all_filings:
        if f["date"].startswith(str(fiscal_year + 1)):
            return f
    for f in all_filings:
        if f["date"].startswith(str(fiscal_year)):
            return f
    return None


#building doc url- aig's cik (5272) in the archives path
def get_doc_url(filing):
    acc      = filing["accession"]       
    acc_fmt  = filing["accession_fmt"]  
    base     = f"{EDGAR_WWW}/Archives/edgar/data/{AIG_CIK_INT}/{acc}"

    #using primaryDocument from submissions JSON
    primary = filing.get("primary_doc", "")
    if primary and primary.endswith((".htm", ".txt", ".html")):
        return f"{base}/{primary}"

    #fetching the human-readable index page and parse links
    try:
        index_url = f"{EDGAR_WWW}/Archives/edgar/data/{AIG_CIK_INT}/{acc}/{acc_fmt}-index.htm"
        req = urllib.request.Request(index_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            html = r.read().decode("latin-1")
        # Find .htm links — the 10-K body is usually the largest
        import re
        links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"', html)
        if links:
            return f"https://www.sec.gov{links[0]}"
    except Exception as e:
        print(f"\n    [index.htm fallback failed: {e}]")

    #fetching JSON directory listing from data.sec.gov
    try:
        dir_url  = f"{EDGAR_DATA}/Archives/edgar/data/{AIG_CIK_INT}/{acc}/{acc_fmt}-index.json"
        idx      = fetch_json(dir_url)
        items    = idx.get("directory", {}).get("item", [])
        candidates = sorted(
            [(int(i.get("size", 0)), i["name"]) for i in items
             if i.get("name","").endswith((".htm",".txt")) and int(i.get("size",0)) > 50_000],
            reverse=True
        )
        if candidates:
            return f"{base}/{candidates[0][1]}"
    except Exception as e:
        print(f"\n    [JSON index fallback failed: {e}]")

    return None


def run():
    print(f"\n{'='*60}")
    print(f" Downloading {COMPANY_NAME} 10-K from SEC EDGAR")
    print(f" Years: {YEARS}  |  Output: {RAW_DIR}")
    print(f"{'='*60}\n")

    all_filings = all_10k_filings()
    downloaded  = []

    for year in YEARS:
        out = RAW_DIR / f"{COMPANY_NAME}_{year}.txt"
        if out.exists() and out.stat().st_size > 10_000:
            print(f"  [SKIP] {out.name} ({out.stat().st_size:,} bytes)")
            downloaded.append(str(out)); continue

        filing = find_filing_for_year(all_filings, year)
        if not filing:
            print(f"  [WARN] No 10-K for fiscal year {year}"); continue

        print(f"  fiscal_year={year}  acc={filing['accession_fmt']}  filed={filing['date']}")
        print(f"  primary_doc={filing['primary_doc']}")

        try:
            doc_url = get_doc_url(filing)
            if not doc_url:
                print("  [WARN] Could not locate document file"); continue
            text = fetch_text(doc_url)
            out.write_text(text, encoding="utf-8")
            print(f"done  ({len(text):,} chars  →  {out.name})")
            downloaded.append(str(out))
        except Exception as e:
            print(f"\n  [ERROR] {e}")

        time.sleep(0.5)

    (RAW_DIR / "manifest.json").write_text(json.dumps({"files": downloaded}, indent=2))
    print(f"\n✓  {len(downloaded)} filing(s) saved.\n")
    return downloaded

if __name__ == "__main__":
    run()
