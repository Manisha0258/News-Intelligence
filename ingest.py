"""
ingest.py
Fetches news articles. Feed URLs are auto-discovered from each site's
homepage instead of being hand-typed — most news sites declare their RSS
feed in a hidden <link> tag, even when there's no visible "RSS" button.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import feedparser
from sentence_transformers import SentenceTransformer
import chromadb

# Just homepages now — no more hunting for exact feed URLs by hand.
NEWS_SITES = {
    "BBC": "https://www.bbc.com/news",
    "NPR": "https://www.npr.org",
    "Reuters": "https://www.reuters.com",
    "TechCrunch": "https://techcrunch.com",
}

DB_PATH = "./chroma_db"
COLLECTION_NAME = "news"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsRAGBot/1.0)"}


def discover_feed_url(homepage_url):
    """Look at a site's homepage HTML and find its declared RSS/Atom feed."""
    try:
        response = requests.get(homepage_url, headers=_HEADERS, timeout=6)
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("link", rel="alternate"):
            feed_type = link.get("type", "")
            if "rss" in feed_type or "atom" in feed_type:
                href = link.get("href")
                if href:
                    return urljoin(homepage_url, href)
    except Exception:
        pass
    return None


def get_rss_feeds():
    """Resolve each configured homepage to an actual feed URL."""
    feeds = {}
    for name, homepage in NEWS_SITES.items():
        feed_url = discover_feed_url(homepage)
        if feed_url:
            feeds[name] = feed_url
    return feeds


def fetch_articles():
    """Pull recent articles from every auto-discovered feed."""
    articles = []
    for source, url in get_rss_feeds().items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": entry.get("link", ""),
                "source": source,
            })
    return articles


def chunk_article(article, max_chars=600):
    text = f"{article['title']}. {article['summary']}"
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def build_index():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    articles = fetch_articles()
    ids, texts, metadatas = [], [], []
    chunk_id = 0
    for article in articles:
        for chunk in chunk_article(article):
            ids.append(str(chunk_id))
            texts.append(chunk)
            metadatas.append({
                "title": article["title"],
                "link": article["link"],
                "source": article["source"],
            })
            chunk_id += 1

    if not texts:
        return 0

    embeddings = model.encode(texts).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(texts)
