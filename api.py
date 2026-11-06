
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://nimble-biscochitos-38eaae.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about restaurants.

Use only the following restaurant reviews to answer the question.
If the reviews do not contain enough information, say so honestly.

Relevant reviews:
{reviews}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Local AI Agent API is running!"}



@app.post("/chat")
def chat(request: ChatRequest):
    reviews = retriever.invoke(request.question)

    review_text = "\n\n".join(
        document.page_content for document in reviews
    )

    result = chain.invoke({
        "reviews": review_text,
        "question": request.question
    })

    return {"answer": result}




