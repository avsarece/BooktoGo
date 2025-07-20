"""Book Data"""
import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
csv_file = BASE_DIR / "data" / "book_data.csv"
csv_file = str(csv_file)

st.set_page_config(page_title="Book Data", page_icon="🐍", layout="wide")
df = pd.read_csv(csv_file, dtype=str)

df_sorted = df.drop(columns=["URL", "Cover_URL", "UPC", "Description"], errors="ignore")
st.title("Book Data")

st.write(df_sorted)
