
from dotenv import load_dotenv
from fastapi import HTTPException
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
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
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
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

    try:
        print("STEP 3: Sending request to Groq", flush=True)

        result = chain.invoke({
            "reviews": review_text,
            "question": request.question
        })

        print("STEP 4: Groq response received", flush=True)

        return {"answer": result.content}

    except Exception as e:
        print(f"Groq error: {str(e)}", flush=True)

        raise HTTPException(
            status_code=502,
            detail="Unable to generate an answer. Please try again later."
        )

   