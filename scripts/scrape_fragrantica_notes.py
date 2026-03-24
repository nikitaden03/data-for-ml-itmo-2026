"""
Scrape notes and ratings from Fragrantica for a list of perfumes.
Uses Playwright non-headless mode to bypass Cloudflare.
Finds correct URLs via search, then scrapes each perfume page.
"""

import json
import time
import random
import sys
import pandas as pd
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

PERFUMES = [
    ("Chanel No 5 Eau de Parfum", "Chanel"),
    ("By the Fireplace", "Maison Margiela"),
    ("Cuoium", "Orto Parisi"),
    ("Dioressence", "Dior"),
    ("Black Pearls", "Elizabeth Taylor"),
    ("Acqua di Parma Colonia", "Acqua di Parma"),
    ("The One Gentleman", "Dolce Gabbana"),
    ("Aspen For Men", "Coty"),
]


def human_delay(min_s=3.0, max_s=7.0):
    time.sleep(random.uniform(min_s, max_s))


def search_perfume_url(page, name: str, brand: str):
    """Search Fragrantica and return best matching URL."""
    query = f"{name} {brand}".replace("&", "").replace(" ", "+")
    url = f"https://www.fragrantica.com/search/?query={query}"
    print(f"  Searching: {url}")
    page.goto(url, timeout=30000, wait_until="domcontentloaded")
    human_delay(4, 7)

    links = page.eval_on_selector_all(
        "a[href*='/perfume/']",
        "els => els.map(e => ({href: e.href, text: e.innerText.trim()}))"
    )

    name_lower = name.lower()
    brand_lower = brand.lower()

    # Score each link by name match
    best_url = None
    best_score = 0
    for item in links:
        href = item.get("href", "")
        text = item.get("text", "").lower()
        if "/perfume/" not in href or not href.endswith(".html"):
            continue
        score = 0
        # Check if key words from name appear in link text or href
        for word in name_lower.split():
            if len(word) > 3 and word in text:
                score += 2
            if len(word) > 3 and word in href.lower():
                score += 1
        for word in brand_lower.split():
            if len(word) > 3 and word in href.lower():
                score += 1
        if score > best_score:
            best_score = score
            best_url = href

    if best_url:
        print(f"  Best match (score={best_score}): {best_url}")
    else:
        # Fallback: first perfume link
        for item in links:
            href = item.get("href", "")
            if "/perfume/" in href and href.endswith(".html"):
                best_url = href
                print(f"  Fallback first link: {best_url}")
                break

    return best_url


def scrape_perfume_page(page, name: str, brand: str, url: str) -> dict:
    """Load a perfume page and extract rating + notes."""
    result = {
        "name": name,
        "brand": brand,
        "url": url,
        "rating": None,
        "votes": None,
        "top_notes": [],
        "middle_notes": [],
        "base_notes": [],
    }

    print(f"  Loading: {url}")
    page.goto(url, timeout=30000, wait_until="domcontentloaded")
    human_delay(5, 8)

    # Rating
    try:
        rating_el = page.query_selector('[itemprop="ratingValue"]')
        if rating_el:
            result["rating"] = float(rating_el.inner_text().strip())
        votes_el = page.query_selector('[itemprop="ratingCount"]')
        if votes_el:
            content = votes_el.get_attribute("content")
            result["votes"] = int(content) if content else None
    except Exception as e:
        print(f"  Rating error: {e}")

    # Notes via pyramid-note-label positioned relative to h4 headings
    try:
        h4_data = page.eval_on_selector_all(
            "h4",
            "els => els.map(e => ({text: e.innerText.trim().toUpperCase(), top: e.getBoundingClientRect().top}))"
        )
        label_data = page.eval_on_selector_all(
            ".pyramid-note-label",
            "els => els.map(e => ({text: e.innerText.trim(), top: e.getBoundingClientRect().top}))"
        )

        note_h4s = sorted(
            [(d["text"], d["top"]) for d in h4_data
             if any(k in d["text"] for k in ["TOP", "MIDDLE", "HEART", "BASE"])],
            key=lambda x: x[1]
        )
        print(f"  Sections: {[h[0] for h in note_h4s]}, Labels: {len(label_data)}")

        def get_section(label_top):
            section = None
            for h4_text, h4_top in note_h4s:
                if label_top >= h4_top - 5:
                    if "TOP" in h4_text:
                        section = "top"
                    elif "MIDDLE" in h4_text or "HEART" in h4_text:
                        section = "middle"
                    elif "BASE" in h4_text:
                        section = "base"
            return section

        for label in label_data:
            text = label["text"]
            if not text or len(text) > 60:
                continue
            section = get_section(label["top"])
            if section == "top":
                result["top_notes"].append(text)
            elif section == "middle":
                result["middle_notes"].append(text)
            elif section == "base":
                result["base_notes"].append(text)

    except Exception as e:
        print(f"  Notes error: {e}")

    return result


def main():
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        # Warm-up
        print("Warming up with homepage...")
        page.goto("https://www.fragrantica.com/", timeout=30000, wait_until="domcontentloaded")
        human_delay(4, 6)

        for name, brand in PERFUMES:
            print(f"\n=== {name} ({brand}) ===")

            # Step 1: Find URL via search
            url = search_perfume_url(page, name, brand)
            if not url:
                print(f"  Could not find URL, skipping.")
                results.append({"name": name, "brand": brand, "url": None, "rating": None, "votes": None,
                                 "top_notes": [], "middle_notes": [], "base_notes": []})
                continue

            human_delay(2, 4)

            # Step 2: Scrape perfume page
            data = scrape_perfume_page(page, name, brand, url)
            results.append(data)

            print(f"  Rating: {data['rating']} ({data['votes']} votes)")
            print(f"  Top:    {data['top_notes']}")
            print(f"  Middle: {data['middle_notes']}")
            print(f"  Base:   {data['base_notes']}")

            human_delay(3, 6)

        browser.close()

    # Save JSON
    with open("data/raw/fragrantica_scraped.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save CSV
    df = pd.DataFrame(results)
    df["top_notes"] = df["top_notes"].apply(lambda x: ", ".join(x) if x else "")
    df["middle_notes"] = df["middle_notes"].apply(lambda x: ", ".join(x) if x else "")
    df["base_notes"] = df["base_notes"].apply(lambda x: ", ".join(x) if x else "")
    df.to_csv("data/raw/fragrantica_scraped.csv", index=False, encoding="utf-8-sig")

    print("\n=== FINAL RESULT ===")
    print(df[["name", "brand", "rating", "votes", "top_notes", "middle_notes", "base_notes"]].to_string())
    print("\nSaved: data/raw/fragrantica_scraped.csv + .json")


if __name__ == "__main__":
    main()
