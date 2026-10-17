"""Numbers for Figure 8 (growth of the literature): yearly PubMed records of the audit query and
their share of all PubMed records (v19, ledger AUD6).

Query counts: work/2026-10-15_growth/counts.csv (Audit chat, branch claude/priceless-tharp-212621,
retrieved 15 Oct 2026), read with git show so that branch need not be merged. Totals: the same
esearch call with term all[sb] (every PubMed record), publication date (pdat), the same October to
September windows. Years overlap slightly: a record with two publication dates can count in both.

Output: results/v19/F8_numbers.csv (year_ending_sep, from, to, records, pubmed_total, share_percent)
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "origin/claude/priceless-tharp-212621:work/2026-10-15_growth/counts.csv"
OUT = ROOT / "results" / "v19" / "F8_numbers.csv"
# Totals the Audit chat reported for three of the windows (AUD6); checked below.
AUDIT_TOTALS = {2016: 1302634, 2021: 1830911, 2026: 1945232}


def count(term: str, start: str, end: str) -> int:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(
        dict(db="pubmed", term=term, rettype="count", datetype="pdat", mindate=start, maxdate=end, retmode="json"))
    for attempt in range(5):
        try:
            return int(json.load(urllib.request.urlopen(url, timeout=60))["esearchresult"]["count"])
        except Exception:
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"esearch failed for {start}-{end}")


def main() -> None:
    text = subprocess.run(["git", "show", SOURCE], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    rows = list(csv.DictReader(io.StringIO(text)))
    for r in rows:
        r["pubmed_total"] = count("all[sb]", r["from"], r["to"])
        r["share_percent"] = round(100 * int(r["records"]) / r["pubmed_total"], 3)
        r["retrieved"] = date.today().isoformat()
        print(r["year_ending_sep"], r["records"], r["pubmed_total"], r["share_percent"], flush=True)
        time.sleep(0.4)
    for year, total in AUDIT_TOTALS.items():
        got = next(r["pubmed_total"] for r in rows if int(r["year_ending_sep"]) == year)
        print(f"check {year}: {got} (Audit {total}, difference {got - total})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year_ending_sep", "from", "to", "records", "pubmed_total",
                                          "share_percent", "retrieved"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
