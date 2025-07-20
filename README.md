# BookRecommendation
A book recommendation system using the website: 'https://books.toscrape.com/'

`data/book_data.csv` and `data/book_data.json` are sample datasets scraped from [books.toscrape.com](http://books.toscrape.com) 
for demonstration purposes only.

### HOW TO RUN THIS PROJECT

### 1. Start Qdrant (Vector DB)

Make sure Docker is running, then run:

```bash
docker run -d \
  --name qdrant123 \
  -p 6333:6333 \
  qdrant/qdrant
```
### 2. Run app_interface/backend/rec_system.py and app_interface/backend/vector_db.py

### 3. Run the system
In terminal, type:
```bash
streamlit run app_interface/app.py 
```
### And you're all done!
