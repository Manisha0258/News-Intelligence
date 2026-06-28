import feedparser
from sentence_transformers import SentenceTransformer
import chromadb

RSS_FEEDS = {
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "Reuters": "https://feeds.reuters.com/reuters/worldNews",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "TechCrunch": "https://techcrunch.com/feed/",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "Guardian Tech": "https://www.theguardian.com/technology/rss",
    "NASA": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
}

DB_PATH = "./chroma_db"
COLLECTION_NAME = "news"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def fetch_articles():
    articles = []
    seen_links = set()
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                link = entry.get("link", "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                articles.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", ""),
                    "link": link,
                    "source": source,
                })
        except Exception:
            continue
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
