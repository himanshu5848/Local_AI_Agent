
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

_retriever = None


def get_retriever():
    global _retriever

    # Reuse the retriever if already initialized
    if _retriever is not None:
        return _retriever

    # Read restaurant reviews
    csv_path = BASE_DIR / "restaurant_reviews.csv"
    df = pd.read_csv(csv_path)

    # Initialize Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    # Select database location
    if Path("/data").exists():
        db_location = "/data/chroma_gemini_db"
    else:
        db_location = str(BASE_DIR / "chroma_gemini_db")

    # Prepare documents
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

    # Initialize ChromaDB
    vector_store = Chroma(
        collection_name="restaurant_reviews_gemini",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    # Add documents only if collection is empty
    existing_ids = vector_store.get()["ids"]

    if not existing_ids:
        vector_store.add_documents(
            documents=documents,
            ids=ids
        )

    # Create and cache retriever
    _retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    return _retriever
