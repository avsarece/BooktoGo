"""App interface
How to run: write streamlit run app_interface/app.py on terminal
"""
import streamlit as st
import pandas as pd
import backend.rec_system as rs
from qdrant_client import QdrantClient
from pathlib import Path

client = QdrantClient(host="localhost", port=6333)
st.set_page_config(page_title="Book Recommender", page_icon="📖", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
csv_file = BASE_DIR / "data" / "book_data.csv"
csv_file = str(csv_file)

df = pd.read_csv(csv_file)

books_list = df["Title"]
book_cover_url = df['Cover_URL']
book_url = df['URL']
st.header('BooktoGo')

selected_book = st.selectbox("Select a book:", books_list.unique())

if st.button("Show Recommend"):
    book_name1 = rs.query_to_qdrant(selected_book)[0]
    book_name2 = rs.query_to_qdrant(selected_book)[1]
    book_name3 = rs.query_to_qdrant(selected_book)[2]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"[**Title:** {book_name1['title']}]({book_name1['url']})")
        st.text(f"Genre: {book_name1['genre']}")
        st.text(f"Rating: {book_name1['rating']}")
        st.image(book_name1['cover_url'])

    with col2:
        st.markdown(f"[**Title:** {book_name2['title']}]({book_name2['url']})")
        st.text(f"Genre: {book_name2['genre']}")
        st.text(f"Rating: {book_name2['rating']}")
        st.image(book_name2['cover_url'])

    with col3:
        st.markdown(f"[**Title:** {book_name3['title']}]({book_name3['url']})")
        st.text(f"Genre: {book_name3['genre']}")
        st.text(f"Rating: {book_name3['rating']}")
        st.image(book_name3['cover_url'])
