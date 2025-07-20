"""Create a Vector db and upload it to qdrant"""
from sentence_transformers import SentenceTransformer
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import PointStruct
import nltk
from pathlib import Path

TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name="book_sentences",
    vectors_config=VectorParams(size=384, distance=Distance.DOT),
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
csv_file = BASE_DIR / "data" / "book_data.csv"
csv_file = str(csv_file)


def upc_to_sentences(csv_file: str) -> dict:
    """Given a csv file, create a dictionary that maps book upc to its corresponding description, split to sentences."""
    df = pd.read_csv(csv_file, header=0)
    upcs = df["UPC"].fillna("").astype(str)

    dictionary = {}
    for upc in upcs:
        description = df.loc[df['UPC'] == upc, 'Description'].fillna("").values[0]
        sentences = split_sentences(description)
        dictionary[upc] = sentences

    return dictionary


def split_sentences(description: str) -> list:
    """Split the sentences, store them in a list and remove the '...more'
    """

    if description == '':
        return ['']

    list1 = TOKENIZER.tokenize(description)
    list1.pop()
    return list1


def remove_more(description: str) -> str:
    substring_to_remove = "...more"
    new_string = description.replace(substring_to_remove, "")
    return new_string


def transform_sentences(csv_file: str, batch_size: int = 64):
    """Transform the book description using SentenceTransformer and upsert to Qdrant"""
    df = pd.read_csv(csv_file, header=0)
    descriptions = df["Description"].fillna("").astype(str)
    genres = df["Genre"].fillna("").astype(str)
    ratings = df["Rating"].fillna(0).astype(int)
    upcs = df["UPC"].fillna("").astype(str)
    titles = df['Title'].fillna("").astype(str)
    urls = df["URL"].fillna("").astype(str)
    cover_urls = df["Cover_URL"].fillna("").astype(str)

    descriptions_lst = []
    desc_upc_genre_rating_title = []  # tuple containing book info

    for desc, upc, genre, rating, title, url, cover_url in zip(descriptions, upcs, genres, ratings, titles, urls,
                                                               cover_urls):
        desc_upc_genre_rating_title.append((desc, upc, genre, rating, title, url, cover_url))
        descriptions_lst.append(remove_more(desc))

    embeddings = model.encode(
        descriptions_lst,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    descp_result = [pair[0] for pair in desc_upc_genre_rating_title]
    upcs_result = [pair[1] for pair in desc_upc_genre_rating_title]
    genres_result = [pair[2] for pair in desc_upc_genre_rating_title]
    ratings_result = [pair[3] for pair in desc_upc_genre_rating_title]
    titles_result = [pair[4] for pair in desc_upc_genre_rating_title]
    urls_result = [pair[5] for pair in desc_upc_genre_rating_title]
    cover_urls_result = [pair[6] for pair in desc_upc_genre_rating_title]

    return descp_result, embeddings, upcs_result, genres_result, ratings_result, titles_result, urls_result, cover_urls_result


def upload_to_qdrant(descps, embeddings, sentence_upcs, genres, ratings, titles, urls, cover_urls, batch_size=100):
    """Uploads sentence embeddings with associated metadata to a Qdrant vector database."""

    points = []
    for idx, (descp, vector, upc, genre, rating, title, url, cover_url) in enumerate(
            zip(descps, embeddings, sentence_upcs, genres, ratings, titles, urls, cover_urls), start=1):
        points.append(PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={
                "title": title,
                "upc": str(upc),
                "genre": genre,
                "descp": descp,
                "rating": rating,
                "url": url,
                "cover_url": cover_url
            }
        ))

        if len(points) >= batch_size:
            client.upsert(collection_name="book_sentences", wait=True, points=points)
            points = []

    if points:
        client.upsert(collection_name="book_sentences", wait=True, points=points)


descps, embeddings, upcs, genres, ratings, titles, urls, cover_urls = transform_sentences(csv_file)
upload_to_qdrant(descps, embeddings, upcs, genres, ratings, titles, urls, cover_urls)

# Retract first 15 data from collection
points, next_page = client.scroll(
    collection_name="book_sentences",
    limit=15,
    with_payload=True,  # Also include other metadata
    with_vectors=True,
)

# Print the points
for point in points:
    print(f"ID: {point.id}")
    print(f"Title     : {point.payload.get('title')}")
    print(f"UPC     : {point.payload.get('upc')}")
    print(f"Genre: {point.payload.get('genre')}")
    print(f"Rating: {point.payload.get('rating')}")
    print(f"Description: {point.payload.get('descp')[:10]}")
    print(f"Vector  : {point.vector[:5]} ...")
    print(f"URL: {point.payload.get('url')}")
    print(f"cover_URL: {point.payload.get('cover_url')}")
    print("---")


def query_to_qdrant(query: list[str]):
    """Send a query to qdrant, and print the closest 3 search results to this query.
    >>> query_to_qdrant(['Traveling to Africa'])
    >>> query_to_qdrant(['A magical world'])
    >>> query_to_qdrant(['Dissociation and isolation and Gregor Samsa'])

    """

    embedding = model.encode(query)
    client = QdrantClient("localhost", port=6333)
    search_result = client.search(
        collection_name="book_sentences",
        query_vector=embedding,
        limit=3
    )
    print(search_result)


def compute_cos_similarities(sentences1: list[str], sentences2: list[str]):
    """Compute the cosine similarity of two embeddings.
    >>>compute_cos_similarities(['The dog ate a cat'], ['The cat ate a dog'])
    """

    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    similarities = model.similarity(embeddings1, embeddings2)

    for idx_i, sentence1 in enumerate(sentences1):
        print(sentence1)
        for idx_j, sentence2 in enumerate(sentences2):
            print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")
