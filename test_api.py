from fastapi.testclient import TestClient
from api import app
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Local AI Agent API is running!"
    }

def test_invalid_route():
    response = client.get("/invalid-route")

    assert response.status_code == 404


@patch("api.get_chain")
@patch("api.get_retriever")
def test_chat(mock_get_retriever, mock_get_chain):

    # Fake restaurant review
    fake_document = SimpleNamespace(
        page_content="The pizza was delicious and fresh."
    )

    # Fake retriever
    mock_get_retriever.return_value.invoke.return_value = [
        fake_document
    ]

    # Fake Gemini response
    fake_response = SimpleNamespace(
        content=[
            {"type": "text", "text": "The pizza was delicious."}
        ]
    )

    mock_get_chain.return_value.invoke.return_value = fake_response

    # Send request
    response = client.post(
        "/chat",
        json={"question": "How was the pizza?"}
    )

    # Verify response
    assert response.status_code == 200
    assert response.json() == {
        "answer": "The pizza was delicious."
    }

def test_chat_missing_question():
    response = client.post(
        "/chat",
        json={}
    )

    assert response.status_code == 422