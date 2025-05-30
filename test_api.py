from fastapi.testclient import TestClient
from chatbot_api import app

client = TestClient(app)

def test_process_endpoint():
    
    # Message
    payload = {"text": "Bonjour, comment ça va aujourd'hui ?"}

    response = client.post("/process/", json=payload)

    # Check response status
    assert response.status_code == 200

    data = response.json()

    # Check returned object
    assert "original" in data
    assert "translation" in data
    assert "sentiment" in data
    assert "response" in data

    # Check returned value
    assert isinstance(data["original"], str)
    assert isinstance(data["translation"], str)
    assert data["sentiment"] in ["Positif", "Neutre", "Négatif"]
    assert isinstance(data["response"], str)