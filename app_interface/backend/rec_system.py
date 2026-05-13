"""Recommendation System"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch import IntTensor
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue
)
from pathlib import Path
import streamlit as st

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

qdrant_client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
csv_file = BASE_DIR / "data" / "book_data.csv"
csv_file = str(csv_file)


def book_to_id(input_song: str) -> str:
    """Create a dictionary that maps every song title to its corresponding id,
    and return the id of the book.
        >>> book_to_id("Viva La Vida")

    """
    csv_path = csv_file
    df = pd.read_csv(csv_path, header=0)
    ids = df["UPC"].fillna("").astype(str)

    dictionary = {}
    for id in ids:
        title = (
            df.loc[df["UPC"] == id, "Title"]
            .fillna("")
            .values[0]
            .strip()
            .lower()
        )
        dictionary[title] = id

    input_song_lower = input_song.strip().lower()
    return dictionary[input_song_lower]


def book_to_upc(csv_file: str, input_book: str) -> str:
    """Given a csv file, create a dictionary that maps every book name to its corresponding book upc,
    and return the upc of the book.
        >>> book_to_upc('book_data.csv', "it's only the himalayas")

    """

    df = pd.read_csv(csv_file, header=0)
    upcs = df["UPC"].fillna("").astype(str)

    dictionary = {}
    for upc in upcs:
        title = (
            df.loc[df["UPC"] == upc, "Title"]
            .fillna("")
            .values[0]
            .strip()
            .lower()
        )
        dictionary[title] = upc

    input_book_altered = input_book.strip().lower()
    return dictionary[input_book_altered]


def transform_genres(csv_file: str, input_genre: str) -> dict:
    """Compute the similarity between the input genre and the genres from the csv file,
    and return the most similar 3 genres to this genre.

    >>> transform_genres('output1.csv', 'academic')
    {'science': 0.5172159671783447, 'biography': 0.4612341821193695, ...}
    """
    df = pd.read_csv(csv_file, header=0)
    genres = df["Genre"].fillna("").astype(str)

    embeddings1 = model.encode(input_genre.lower().strip())
    similarity_genre = {}
    for genre in genres:
        embeddings2 = model.encode(genre)
        similarities = model.similarity(embeddings1, embeddings2)
        similarity_genre[genre.lower()] = IntTensor.item(similarities)  # Turn the tensor with one element to an integer

    # Sort by items
    sorted_genres = sorted(similarity_genre, key=similarity_genre.get, reverse=True)
    sorted_dictionary = {genre: item for genre in sorted_genres for item in [similarity_genre.get(genre, [])]}
    sorted_dictionary.pop(input_genre)
    return sorted_dictionary


def genre_similarity(input_genre: str, genre2: str) -> float:
    """Calculate the similarity of input_genre and genre2"""
    embeddings1 = model.encode(input_genre.lower().strip())
    embeddings2 = model.encode(genre2.lower().strip())

    similarities = model.similarity(embeddings1, embeddings2)
    result = IntTensor.item(similarities)
    return result


def unweighted_similarity(input_book: str) -> list[dict[str, str | int | float]]:
    """Compute the unweighted similarity of the book with other books in qdrant container
    Unweighted similarity only includes the description embeddings.
    >>> unweighted_similarity("it's only the himalayas")
    >>> unweighted_similarity("A Light in the Attic")
    >>> unweighted_similarity("Tipping the Velvet")
    """
    df = pd.read_csv(csv_file, header=0)

    input_upc = book_to_upc(csv_file, input_book)
    book_row = df[df["UPC"] == input_upc]
    book_description = book_row["Description"].values[0]
    query_vec = model.encode([book_description])[0].tolist()

    hits = qdrant_client.search(
        collection_name="book_sentences",
        query_vector=query_vec,
        query_filter=Filter(must_not=[FieldCondition(key="upc", match=MatchValue(value=input_upc))]),
        limit=1000,
        with_payload=['title', 'genre', 'rating'],
        with_vectors=False,
    )

    return [{**hit.payload, 'score': hit.score} for hit in hits]


def weighted_similarity(input_book: str, book2: str) -> float:
    """Compute the weighted similarity of two books, The weighted similarity includes:
    0.6 * text_embedding + 0.25 * genre_embedding + 0.15 * rating_embedding
    If the input_book does not exist... CONTINUE HERE
    >>> weighted_similarity("It's Only the Himalayas", "A Light in the Attic")
    >>> weighted_similarity("The Wedding Dress", 'Shopaholic Ties the Knot (Shopaholic #3)')
    """
    df = pd.read_csv(csv_file, header=0)

    input_upc = book_to_upc(csv_file, input_book)
    input_row = df[df["UPC"] == input_upc]

    match = next((x for x in unweighted_similarity(input_book) if
                  x['title'].strip().lower() == book2.strip().lower()), None)

    if match:
        text_sim = match['score']  # float
        book2_genre = match['genre']  # str
        book2_rating = match['rating']

        input_genre = input_row["Genre"].values[0]
        genre_sim = genre_similarity(input_genre, book2_genre)
        rating_dict = {5: 1, 4: 0.8, 3: 0.6, 2: 0.4, 1: 0.2, 0: 0}

        return 0.6 * text_sim + 0.25 * genre_sim + 0.15 * rating_dict[book2_rating]

    else:
        print("Book not found.")


def find_most_similar(input_book: str) -> list:
    """Find the 3 books that are most similar to this book, using WEIGHTED similarity.
    >>> find_most_similar("It's Only the Himalayas")
    >>> find_most_similar("Tipping the Velvet")
    >>> find_most_similar("How Music Works")
    >>> find_most_similar("The Most Perfect Thing: Inside (and Outside) a Bird's Egg")
    >>> find_most_similar("Something More Than This")
    >>> find_most_similar("The Wedding Dress")
    >>> find_most_similar("The 10% Entrepreneur: Live Your Startup Dream Without Quitting Your Day Job")
    """

    df = pd.read_csv(csv_file, header=0)

    input_upc = book_to_upc(csv_file, input_book)
    input_row = df[df["UPC"] == input_upc]
    if input_row.empty:
        print(f"Book '{input_book}' not found in database.")
        return []

    book_description = input_row["Description"].values[0]
    query_vec = model.encode([book_description])[0].tolist()

    hits = qdrant_client.search(
        collection_name="book_sentences",
        query_vector=query_vec,
        query_filter=Filter(must_not=[FieldCondition(key="upc", match=MatchValue(value=input_upc))]),
        limit=30,
        with_payload=['title', 'genre', 'rating', 'url', 'cover_url'],
        with_vectors=False
        #score_threshold=0.4
    )

    scored_books = []
    for hit in hits:
        similarity = weighted_similarity(input_book, hit.payload['title'])
        if similarity is not None:
            scored_books.append({
                'title': hit.payload['title'],
                'genre': hit.payload['genre'],
                'rating': hit.payload['rating'],
                'score': similarity,
                'URL': hit.payload['url'],
                'Cover URL': hit.payload['cover_url']

            })

    return sorted(scored_books, key=lambda x: -x['score'])[:3]


def query_to_qdrant(book_name: str):
    """Send a query to qdrant, and print the closest 3 search results to this query.
    >>> query_to_qdrant("It's Only the Himalayas")
    >>> query_to_qdrant("Tipping the Velvet")
    >>> query_to_qdrant("How Music Works")
    >>> query_to_qdrant("The Most Perfect Thing: Inside (and Outside) a Bird's Egg")
    >>> query_to_qdrant("Something More Than This")
    >>> query_to_qdrant("The Wedding Dress")
    >>> query_to_qdrant("The 10% Entrepreneur: Live Your Startup Dream Without Quitting Your Day Job")
    """
    df = pd.read_csv(csv_file, header=0)
    input_upc = book_to_id(book_name)
    book_row = df[df["UPC"] == input_upc]
    book_description = book_row['Description'].values[0]
    book_genre = book_row['Genre'].values[0]
    book_rating = book_row['Rating'].values[0]

    embedding = model.encode(book_description)
    search_result = qdrant_client.search(
        collection_name="book_sentences",
        query_vector=embedding,
        query_filter=Filter(must_not=[FieldCondition(key="upc", match=MatchValue(value=input_upc))]),
        with_payload=True,
        limit=30
    )

    dict = {5: 1, 4: 0.8, 3: 0.6, 2: 0.4, 1: 0.2, 0: 0}
    rating_score = 0.25 * dict[book_rating]
    scored_songs = [
        {**hit.payload,
         'score': 0.6 * hit.score + 0.15 * genre_similarity(book_genre, hit.payload['genre']) + rating_score}
        for hit in search_result
    ]
    return sorted(scored_songs, key=lambda x: -x['score'])[:3]
