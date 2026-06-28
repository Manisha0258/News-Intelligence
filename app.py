"""
app.py
The user-facing app. Run locally with: streamlit run app.py
"""

import time
import streamlit as st

import ingest
import database
from retrieve import answer_with_rag, answer_without_rag, _get_embedding_model
from recommender import rank_feed

st.set_page_config(page_title="News RAG Assistant", page_icon="📰", layout="centered")
database.init_db()


@st.cache_data(show_spinner=False)
def load_articles_and_embeddings():
    raw_articles = ingest.fetch_articles()

    seen_links = set()
    unique_articles = []
    for a in raw_articles:
        if a["link"] not in seen_links:
            seen_links.add(a["link"])
            unique_articles.append(a)

    model = _get_embedding_model()
    embeddings, published_times = {}, {}
    fetch_time = time.time()
    for a in unique_articles:
        text = f"{a['title']}. {a['summary']}"
        embeddings[a["link"]] = model.encode(text)
        published_times[a["link"]] = fetch_time

    return unique_articles, embeddings, published_times


st.title("📰 News RAG Assistant")

with st.sidebar:
    st.header("Index")
    if st.button("🔄 Refresh news index"):
        with st.spinner("Fetching and embedding articles..."):
            ingest.build_index()
        load_articles_and_embeddings.clear()
        st.success("Index refreshed!")
    st.caption("First load takes a little longer while the embedding model loads.")

tab1, tab2 = st.tabs(["🔁 Swipe feed", "💬 Ask AI"])

with tab1:
    articles, embeddings, published_times = load_articles_and_embeddings()
    feed = rank_feed(articles, embeddings, published_times)

    if not feed:
        st.info("No more articles to show — refresh the index from the sidebar.")
    else:
        article = feed[0]
        st.subheader(article["title"])
        st.write(article["summary"])
        st.caption(f"Source: {article['source']}")
        st.markdown(f"[Read full article]({article['link']})")

        col1, col2, col3 = st.columns(3)
        if col1.button("👍 Like"):
            database.log_interaction(article["link"], "like")
            st.rerun()
        if col2.button("👎 Dislike"):
            database.log_interaction(article["link"], "dislike")
            st.rerun()
        if col3.button("⭐ Save"):
            database.log_interaction(article["link"], "save")
            st.rerun()

with tab2:
    question = st.text_input("Ask a question about recent news:")
    show_comparison = st.checkbox("Show without-RAG comparison")

    if st.button("Ask") and question:
        st.subheader("✅ With RAG")
        with st.spinner("Retrieving and answering..."):
            answer, sources = answer_with_rag(question)
        st.write(answer)
        if sources:
            st.markdown("**Sources:**")
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['link']}) — *{s['source']}*")

        if show_comparison:
            st.subheader("❌ Without RAG")
            with st.spinner("Asking without retrieval..."):
                plain = answer_without_rag(question)
            st.write(plain)
