"""
ingest.py
Fetches news articles using Google News' RSS search, which aggregates
results from many different publishers per topic — no need to maintain
a list of individual site feed URLs at all.
"""

import feedparser
from sentence_transformers import SentenceTransformer
import chromadb

NEWS_TOPICS = [
    "world news",
    "technology",
    "business",
    "science",
    "health",
    "sports",
]

DB_PATH = "./chroma_db"
COLLECTION_NAME = "news"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _google_news_rss_url(topic):
    query = topic.replace(" ", "+")
    return f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def fetch_articles():
    articles = []
    seen_links = set()

    for topic in NEWS_TOPICS:
        feed = feedparser.parse(_google_news_rss_url(topic))
        for entry in feed.entries:
            link = entry.get("link", "")
            if not link or link in seen_links:
                continue
            seen_links.add(link)

            source_name = "Google News"
            if hasattr(entry, "source") and hasattr(entry.source, "title"):
                source_name = entry.source.title

            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "link": link,
                "source": source_name,
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
