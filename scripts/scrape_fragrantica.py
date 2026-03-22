"""
Scraping perfume data from Fragrantica
Brands: Tom Ford (20), Creed (20)
Columns match Kaggle dataset: url, perfume, brand, country, gender,
  rating_value, rating_count, year, top, middle, base,
  perfumer1, perfumer2, mainaccord1..5
"""

import time
import re
import sys
import io
import pandas as pd
from pathlib import Path
from curl_cffi import requests as cf_requests
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BRAND_URLS = {
    "Tom Ford": "https://www.fragrantica.com/designers/Tom-Ford.html",
    "Creed":    "https://www.fragrantica.com/designers/Creed.html",
}

session = cf_requests.Session(impersonate="chrome120")
SLEEP = 1.5


def get_soup(url: str) -> BeautifulSoup:
    r = session.get(url, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")


def parse_rating(soup: BeautifulSoup) -> tuple:
    rating = soup.select_one('span[itemprop="ratingValue"]')
    count  = soup.select_one('span[itemprop="ratingCount"]')
    return (
        rating.text.strip() if rating else "",
        count.text.strip().replace(",", "") if count else "",
    )


def parse_notes(soup: BeautifulSoup) -> tuple:
    levels = soup.find_all("div", class_="pyramid-level-container")
    result = []
    for level in levels:
        notes = [
            img["alt"]
            for a in level.find_all("a", class_="pyramid-note-link")
            if (img := a.find("img")) and img.get("alt")
        ]
        result.append(notes)
    while len(result) < 3:
        result.append([])
    return result[0], result[1], result[2]


def parse_accords(soup: BeautifulSoup) -> list:
    return [s.text.strip() for s in soup.select("span.accord-bar") if s.text.strip()]


def parse_perfumers(soup: BeautifulSoup) -> tuple:
    perfumers = []
    for a in soup.select('a[href*="/noses/"]'):
        name = a.get_text(strip=True)
        if name:
            perfumers.append(name)
    perfumers = list(dict.fromkeys(perfumers))  # deduplicate, keep order
    p1 = perfumers[0] if len(perfumers) > 0 else ""
    p2 = perfumers[1] if len(perfumers) > 1 else ""
    return p1, p2


def parse_country(soup: BeautifulSoup) -> str:
    # Country usually appears near brand info
    for a in soup.select('a[href*="/country/"], a[href*="/designers/"]'):
        text = a.get_text(strip=True)
        # Heuristic: short text after brand section
        if text and len(text) < 40 and text not in ("Tom Ford", "Creed"):
            pass
    # Try meta description
    desc = soup.find("meta", {"name": "description"})
    if desc:
        content = desc.get("content", "")
        m = re.search(r'from ([A-Z][a-z]+(?: [A-Z][a-z]+)?)', content)
        if m:
            return m.group(1)
    # Try structured data
    text = soup.get_text()
    for pattern in [r'Country of origin[:\s]+([A-Za-z ]+)', r'Origin[:\s]+([A-Za-z ]+)']:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""


def parse_gender(soup: BeautifulSoup) -> str:
    # gender shown in breadcrumb or specific div
    for sel in ['span[itemprop="description"]', 'p.breadcrumb']:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(strip=True).lower()
            for g in ["for women and men", "for women", "for men", "unisex"]:
                if g in t:
                    return g
    # fallback: og:description
    meta = soup.find("meta", {"property": "og:description"})
    if meta:
        content = meta.get("content", "").lower()
        for g in ["for women and men", "for women", "for men", "unisex"]:
            if g in content:
                return g
    return ""


def parse_year(soup: BeautifulSoup) -> str:
    text = soup.get_text()
    m = re.search(r'launched in (\d{4})', text, re.I)
    if m:
        return m.group(1)
    # Also check title/h1 area
    m2 = re.search(r'\b(19|20)\d{2}\b', soup.find("h1").get_text() if soup.find("h1") else "")
    if m2:
        return m2.group(0)
    return ""


def get_brand_perfume_links(brand_url: str, top_n: int = 20) -> list:
    print(f"  Fetching brand page: {brand_url}")
    soup = get_soup(brand_url)
    time.sleep(SLEEP)

    perfume_links = []
    seen = set()
    for a in soup.select("a[href*='/perfume/']"):
        href = a.get("href", "")
        if href in seen:
            continue
        if not re.search(r"/perfume/[^/]+/[^/]+-\d+\.html$", href):
            continue
        seen.add(href)

        # Make absolute URL
        if href.startswith("/"):
            href = "https://www.fragrantica.com" + href

        text = a.get_text(strip=True)
        m = re.search(r"(\d{3,})\s*$", text)
        count = int(m.group(1)) if m else 0
        perfume_links.append({"url": href, "rating_count_approx": count})

    perfume_links.sort(key=lambda x: x["rating_count_approx"], reverse=True)
    return perfume_links[:top_n]


def scrape_perfume(url: str, brand: str):
    try:
        soup = get_soup(url)
        time.sleep(SLEEP)

        name_tag = soup.find("h1", itemprop="name") or soup.find("h1")
        name_raw = name_tag.get_text(strip=True) if name_tag else url.split("/")[-1]
        # h1 contains "Perfume Name  Brand\nfor women" — strip brand and gender
        for suffix in [brand, "for women and men", "for women", "for men", "unisex"]:
            name_raw = name_raw.replace(suffix, "").strip()
        name = name_raw.strip()

        rating_value, rating_count = parse_rating(soup)
        top, middle, base = parse_notes(soup)
        accords = parse_accords(soup)
        perfumer1, perfumer2 = parse_perfumers(soup)
        country = parse_country(soup)
        gender = parse_gender(soup)
        year = parse_year(soup)

        def accord(i):
            return accords[i] if i < len(accords) else ""

        return {
            "url":          url,
            "perfume":      name,
            "brand":        brand,
            "country":      country,
            "gender":       gender,
            "rating_value": rating_value,
            "rating_count": rating_count,
            "year":         year,
            "top":          ", ".join(top),
            "middle":       ", ".join(middle),
            "base":         ", ".join(base),
            "perfumer1":    perfumer1,
            "perfumer2":    perfumer2,
            "mainaccord1":  accord(0),
            "mainaccord2":  accord(1),
            "mainaccord3":  accord(2),
            "mainaccord4":  accord(3),
            "mainaccord5":  accord(4),
        }
    except Exception as e:
        print(f"  ERROR scraping {url}: {e}")
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────
all_records = []

for brand, brand_url in BRAND_URLS.items():
    print(f"\n{'='*50}")
    print(f"Brand: {brand}")
    links = get_brand_perfume_links(brand_url, top_n=20)
    print(f"  Top-20 links found: {len(links)}")

    for i, link in enumerate(links):
        label = link["url"].split("/")[-1]
        print(f"  [{i+1:02d}/{len(links)}] {label}")
        rec = scrape_perfume(link["url"], brand)
        if rec:
            all_records.append(rec)
            print(f"        rating={rec['rating_value']} top=[{rec['top'][:40]}]")

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(all_records)
output_path = DATA_DIR / "tom_ford_creed.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n{'='*50}")
print(f"Saved: {output_path}")
print(f"Rows: {len(df)} | Cols: {len(df.columns)}")
print(f"\nBrand breakdown:")
print(df["brand"].value_counts().to_string())
print(f"\nNulls:")
print(df.isna().sum().to_string())
print(f"\nSample (perfume, rating, top):")
print(df[["perfume", "brand", "rating_value", "top"]].to_string())
