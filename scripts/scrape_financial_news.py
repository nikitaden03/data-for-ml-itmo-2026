"""
Scraper for Russian financial news from public RSS feeds.
Sources: smart-lab.ru, rbc.ru
"""
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def parse_rss(url: str, source_name: str, limit: int = 20) -> list[dict]:
    """Parse RSS feed and return list of news dicts."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'xml')
    except Exception as e:
        print(f"[{source_name}] ERROR: {e}")
        return []

    items = soup.find_all('item')[:limit]
    results = []
    for item in items:
        title = item.find('title')
        description = item.find('description')
        pub_date = item.find('pubDate')
        link = item.find('link')

        title_text = title.get_text(strip=True) if title else ''
        body_text = ''
        if description:
            # Strip HTML tags from description
            desc_soup = BeautifulSoup(description.get_text(), 'lxml')
            body_text = desc_soup.get_text(strip=True)

        date_str = ''
        if pub_date:
            raw = pub_date.get_text(strip=True)
            try:
                # RFC 2822 format
                dt = datetime.strptime(raw[:25], '%a, %d %b %Y %H:%M:%S')
                date_str = dt.strftime('%Y-%m-%d')
            except Exception:
                date_str = raw[:10]

        if title_text:
            results.append({
                'title': title_text,
                'body': body_text,
                'date': date_str,
                'source': source_name,
            })

    print(f"[{source_name}] Parsed {len(results)} items")
    return results


def main():
    all_news = []

    # Source 1: smart-lab.ru
    smart_lab = parse_rss(
        'https://smart-lab.ru/rss/',
        'smart_lab_scraped',
        limit=20
    )
    all_news.extend(smart_lab)

    # Source 2: rbc.ru full RSS
    rbc = parse_rss(
        'https://rssexport.rbc.ru/rbcnews/news/30/full.rss',
        'rbc_scraped',
        limit=20
    )
    all_news.extend(rbc)

    df = pd.DataFrame(all_news)
    df = df.drop_duplicates(subset='title')
    df = df[df['title'].str.len() > 10].reset_index(drop=True)

    output_path = 'data/raw/scraped_financial_news.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nTotal scraped: {len(df)} articles")
    print(f"Saved to: {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print("\nSample titles:")
    for t in df['title'].head(5):
        print(f"  - {t}")


if __name__ == '__main__':
    main()
