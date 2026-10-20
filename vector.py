from langchain_ollama import OllamaEmbeddings #  It converts text into a list of numbers called an embedding vector. These numbers represent semantic information about the text.

from langchain_chroma import Chroma #  Chroma is a vector database. It stores text documents and their embeddings so you can search for relevant information using vector similarity.

from langchain_core.documents import Document #It provides a standard structure for storing a piece of text along with its additional information, called metadata. WE CREATE DOCUMENT AND THEN GONNA PASS TO THE CHROMA DATBASE.

import os # os is a built-in Python module for interacting with your operating system.

import pandas as pd # Pandas is a Python library for working with structured data, especially tables. HERE WE USE IT TO READ A CSV FILE EASILY.

df = pd.read_csv("restaurant_reviews.csv") #  This line reads the restaurant_reviews.csv file and stores its data in a Pandas DataFrame (table) called df.

embeddings= OllamaEmbeddings(model="mxbai-embed-large")  #  This line loads the mxbai-embed-large embedding model through Ollama to convert text into numerical vectors that represent its meaning.

db_location = "./chroma_langchain_db" # specifing the location where we want to store our vector database.

add_documents = not os.path.exists(db_location) # This line checks whether the folder or path stored in db_location does not exist; if it doesn’t exist, add_documents becomes True, otherwise it becomes False.

if add_documents:
  documents = []
  ids = []

# preparing the data in the documents ..

  for i, row in df.iterrows():  #  This line loops through each row in the DataFrame df, storing the row’s index in i and its data in row.
    document = Document(

      page_content=row["Title"] + " " + row["Review"], # This line combines the restaurant review's Title and Review text into one string, separated by a space, and assigns it to page_content. ANY OF THE CONTENT THAT WE WANT TO USE TO LOOK UP THE INFORAMTION IN THE DATABASE THAT NEED TO GO IN THE PAGE_CONTENT.

      metadata ={"ratings":row["Rating"],"date": row["Date"]}, #  This line stores the review's rating and date in a dictionary called metadata as additional information about that review. WE WON'T BE QUERYING BASED ON THE MATADATA.

      id=str(i) # This line converts the index i into a string and stores it in id, so each document can have a unique identifier, such as "0", "1", or "2".

    )
    ids.append(str(i))
    documents.append(document)

# add this documents in the vector store ..

vector_store = Chroma(  #  This line starts creating a Chroma vector database object and assigns it to vector_store, which you can use to store and search document embeddings.
   
    collection_name="restaurant_reviews",  #  This line sets the name of the Chroma database collection to "restaurant_reviews", where your restaurant review documents and their embeddings will be organized.

    persist_directory=db_location, #  This line specifies the folder path stored in db_location where Chroma saves the vector database on your computer so it can be reused later.


    embedding_function=embeddings  #   This line tells Chroma to use the embeddings model to convert documents and search queries into numerical vectors for similarity search.

)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # This code converts vector_store into a retriever that searches the database and returns the 5 most relevant documents for a given query (k=5) to the LLM.
)
