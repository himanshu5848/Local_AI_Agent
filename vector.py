
import os
import time
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

    vector_store = Chroma(
        collection_name="restaurant_reviews_gemini",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    # Find reviews that are not already stored
    existing_ids = set(vector_store.get()["ids"])

    missing_documents = []
    missing_ids = []

    for document, doc_id in zip(documents, ids):
        if doc_id not in existing_ids:
            missing_documents.append(document)
            missing_ids.append(doc_id)

    print("Reviews already stored:", len(existing_ids))
    print("Reviews remaining:", len(missing_documents))

    # Add missing reviews in small batches
    batch_size = 5

    for start in range(0, len(missing_documents), batch_size):
        batch_documents = missing_documents[start:start + batch_size]
        batch_ids = missing_ids[start:start + batch_size]

        for attempt in range(5):
            try:
                vector_store.add_documents(
                    documents=batch_documents,
                    ids=batch_ids
                )
                print("Indexed batch:", batch_ids)
                break

            except Exception as e:
                if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                    raise

                if attempt == 4:
                    raise

                wait_time = 10 * (attempt + 1)
                print("Gemini rate limit. Waiting:", wait_time, "seconds")
                time.sleep(wait_time)

        time.sleep(2)

    print("Total stored reviews:", len(vector_store.get()["ids"]))

    _retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    return _retriever