"""Testing"""
from app_interface.backend.vector_db import upload_to_qdrant, transform_sentences
from app_interface.backend.rec_system import find_most_similar, unweighted_similarity


csv_path = 'data/book_data.csv'

if __name__ == "__main__":
    descps, embeddings, upcs, genres, ratings, titles, urls, cover_urls = transform_sentences(csv_path)
    upload_to_qdrant(descps, embeddings, upcs, genres, ratings, titles, urls, cover_urls)

    input_book = input("Enter a book title: ").strip().lower()
    unweighted_similarity(input_book)
    results = find_most_similar(input_book)

    if results:
        print("\nTop 3 similar books:")
        for i, book in enumerate(results, start=1):
            print(f"{i}. {book['title']} — Genre: {book['genre']} | Rating: {book['rating']}")
            print(f"   URL: {book['URL']}")
            print(f"   Cover: {book['Cover URL']}")
            print(f"   Score: {book['score']}\n")
    else:
        print("No similar books found.")
