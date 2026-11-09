
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load the API key from the local .env file
load_dotenv()
print("GOOGLE_API_KEY detected:", bool(os.getenv("GOOGLE_API_KEY")))

# Read restaurant reviews
df = pd.read_csv("restaurant_reviews.csv")

# Google embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# Use Railway Volume in production, local folder otherwise
if Path("/data").exists():
    db_location = "/data/chroma_gemini_db"
else:
    db_location = "./chroma_gemini_db"

# Prepare restaurant reviews as documents
documents = []
ids = []

for i, row in df.iterrows():
    document = Document(
        page_content=str(row["Title"]) + " " + str(row["Review"]),
        metadata={
            "ratings": row["Rating"],
            "date": str(row["Date"])
        },
        id=str(i)
    )

    documents.append(document)
    ids.append(str(i))

# Create or connect to the ChromaDB collection
vector_store = Chroma(
    collection_name="restaurant_reviews_gemini",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents only if the collection is empty
existing_ids = vector_store.get()["ids"]

if not existing_ids:
    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
