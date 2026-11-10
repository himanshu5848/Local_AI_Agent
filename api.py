
from dotenv import load_dotenv
import time
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from vector import get_retriever


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

def get_chain():
    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash"
    )

    prompt = ChatPromptTemplate.from_template(template)

    return prompt | model


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Local AI Agent API is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    print("STEP 1: Chat request received", flush=True)

    retriever = get_retriever()
    chain = get_chain()

    reviews = retriever.invoke(request.question)
    print("STEP 2: Reviews retrieved", len(reviews), flush=True)

    review_text = "\n\n".join(
        document.page_content for document in reviews
    )

    for attempt in range(3):
        try:
            print("STEP 3: Sending request to Gemini", flush=True)

            result = chain.invoke({
                "reviews": review_text,
                "question": request.question
            })
            print("STEP 4: Gemini response received", flush=True)
            break

        except Exception as e:
            if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                raise

            if attempt == 2:
                raise

            wait_time = 5 * (attempt + 1)
            print(f"Gemini unavailable. Retrying in {wait_time} seconds...", flush=True)
            time.sleep(wait_time)

    answer = "".join(
        block["text"]
        for block in result.content
        if block.get("type") == "text"
    )

    return {"answer": answer}