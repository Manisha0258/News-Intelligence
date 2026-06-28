"""
recommender.py
Scores and ranks articles using interest similarity, trending, and freshness.
"""

import time
import numpy as np

from database import get_liked_links, get_seen_links


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compute_trending_scores(article_embeddings, threshold=0.75):
    links = list(article_embeddings.keys())
    scores = {link: 0 for link in links}
    for i in range(len(links)):
        for j in range(len(links)):
            if i != j and cosine_similarity(article_embeddings[links[i]], article_embeddings[links[j]]) > threshold:
                scores[links[i]] += 1
    return scores


def build_interest_vector(article_embeddings):
    liked_links = get_liked_links()
    liked_embeddings = [article_embeddings[link] for link in liked_links if link in article_embeddings]
    if not liked_embeddings:
        return None
    return np.mean(liked_embeddings, axis=0)


def freshness_score(published_time, decay_hours=48):
    age_hours = (time.time() - published_time) / 3600
    return max(0, 1 - age_hours / decay_hours)


def score_article(article, article_embeddings, published_times, interest_vector, trending_scores, max_trending):
    link = article["link"]
    similarity = (
        cosine_similarity(article_embeddings[link], interest_vector)
        if interest_vector is not None else 0.5
    )
    trending = trending_scores.get(link, 0) / max(max_trending, 1)
    fresh = freshness_score(published_times.get(link, time.time()))

    personalized_score = (similarity + trending + fresh) / 3
    essential_score = (trending + fresh) / 2
    return 0.7 * personalized_score + 0.3 * essential_score


def rank_feed(articles, article_embeddings, published_times):
    trending_scores = compute_trending_scores(article_embeddings)
    max_trending = max(trending_scores.values()) if trending_scores else 1
    interest_vector = build_interest_vector(article_embeddings)

    seen = get_seen_links()
    candidates = [a for a in articles if a["link"] not in seen]
    scored = [
        (score_article(a, article_embeddings, published_times, interest_vector, trending_scores, max_trending), a)
        for a in candidates
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored]
