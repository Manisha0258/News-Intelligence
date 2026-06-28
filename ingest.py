import feedparser
from sentence_transformers import SentenceTransformer
import chromadb

RSS_FEEDS = {
    # World News
    "BBC News": "http://feeds.bbci.co.uk/news/rss.xml",
    "Reuters World": "https://feeds.reuters.com/reuters/worldNews",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Guardian World": "https://www.theguardian.com/world/rss",
    "NPR News": "https://feeds.npr.org/1001/rss.xml",
    "ABC News": "https://abcnews.go.com/abcnews/topstories",
    "CBS News": "https://www.cbsnews.com/latest/rss/main",
    "Time Magazine": "https://time.com/feed/",
    "The Hill": "https://thehill.com/feed/",
    "Vox": "https://www.vox.com/rss/index.xml",

    # Technology
    "BBC Technology": "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "TechCrunch": "https://techcrunch.com/feed/",
    "Guardian Tech": "https://www.theguardian.com/technology/rss",
    "Wired": "https://www.wired.com/feed/rss",
    "The Verge": "https://www.theverge.com/rss/index.xml",
    "Ars Technica": "http://feeds.arstechnica.com/arstechnica/index",
    "MIT Tech Review": "https://www.technologyreview.com/feed/",
    "Engadget": "https://www.engadget.com/rss.xml",
    "ZDNet": "https://www.zdnet.com/news/rss.xml",
    "Gizmodo": "https://gizmodo.com/rss",
    "CNET": "https://www.cnet.com/rss/news/",
    "VentureBeat": "https://venturebeat.com/feed/",
    "Hacker News": "https://hnrss.org/frontpage",

    # AI Specific
    "Google AI Blog": "http://googleaiblog.blogspot.com/atom.xml",
    "OpenAI Blog": "https://openai.com/blog/rss/",
    "Towards Data Science": "https://towardsdatascience.com/feed",

    # Business & Finance
    "BBC Business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Guardian Business": "https://www.theguardian.com/business/rss",
    "Forbes": "https://www.forbes.com/real-time/feed2/",
    "Fortune": "https://fortune.com/feed/",
    "Fast Company": "https://www.fastcompany.com/latest/rss",
    "Inc Magazine": "https://www.inc.com/rss/",
    "Business Insider": "https://feeds.businessinsider.com/custom/all",

    # Science
    "NASA": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "Science Daily": "https://www.sciencedaily.com/rss/all.xml",
    "New Scientist": "https://www.newscientist.com/feed/home/",
    "BBC Science": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "Guardian Science": "https://www.theguardian.com/science/rss",
    "Live Science": "https://www.livescience.com/feeds/all",
    "Space.com": "https://www.space.com/feeds/all",
    "Phys.org": "https://phys.org/rss-feed/",

    # Health
    "BBC Health": "http://feeds.bbci.co.uk/news/health/rss.xml",
    "Guardian Health": "https://www.theguardian.com/society/health/rss",
    "Medical News Today": "https://www.medicalnewstoday.com/rss",
    "WebMD": "https://rssfeeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
    "Health Day": "https://consumer.healthday.com/rss",

    # Sports
    "BBC Sport": "http://feeds.bbci.co.uk/sport/rss.xml",
    "Guardian Sport": "https://www.theguardian.com/sport/rss",
    "ESPN": "https://www.espn.com/espn/rss/news",

    # Entertainment & Culture
    "Guardian Culture": "https://www.theguardian.com/culture/rss",
    "Rolling Stone": "https://www.rollingstone.com/feed/",
    "Variety": "https://variety.com/feed/",

    # Politics
    "Politico": "https://www.politico.com/rss/politicopicks.xml",
    "The Atlantic": "https://feeds.feedburner.com/TheAtlantic",
    "BBC Politics": "http://feeds.bbci.co.uk/news/politics/rss.xml",

    # Environment
    "Guardian Environment": "https://www.theguardian.com/environment/rss",
    "BBC Environment": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
}

DB_PATH = "./chroma_db"
COLLECTION_NAME = "news"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def fetch_articles():
    articles = []
    seen_links = set()
    total_feeds = len(RSS_FEEDS)
    successful_feeds = 0

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                link = entry.get("link", "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                title = entry.get("title", "").strip()
                summary = entry.get("summary", "").strip()
                if not title:
                    continue
                articles.append({
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "source": source,
                    "published": entry.get("published", ""),
                })
                count += 1
            if count > 0:
                successful_feeds += 1
        except Exception:
            continue

    return articles


def chunk_article(article, max_chars=600):
    """
    Split article into chunks. Uses both title+summary combined,
    and if summary is long, splits into multiple chunks so no
    information is lost and chunk count stays high.
    """
    text = f"{article['title']}. {article['summary']}"
    chunks = []
    for i in range(0, max(1, len(text)), max_chars):
        chunk = text[i:i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print("Fetching articles from all feeds...")
    articles = fetch_articles()
    print(f"Fetched {len(articles)} unique articles.")

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
                "published": article.get("published", ""),
            })
            chunk_id += 1

    if not texts:
        return 0

    print(f"Embedding {len(texts)} chunks — this may take a minute...")

    # Embed in batches so Streamlit Cloud doesn't time out
    batch_size = 128
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(model.encode(batch).tolist())

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Index built: {len(texts)} chunks from {len(articles)} articles.")
    return len(texts)
