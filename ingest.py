"""
ingest.py
Fetches news articles from RSS feeds, chunks them, generates embeddings,
and stores everything in a local ChromaDB collection.
"""

import feedparser
from sentence_transformers import SentenceTransformer
import chromadb

RSS_FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "Reuters World": "https://feeds.reuters.com/reuters/worldNews",
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
}

DB_PATH = "./chroma_db"
COLLECTION_NAME = "news"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def fetch_articles():
    articles = []
    for source, url in RSS_FEEDS.items():
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
