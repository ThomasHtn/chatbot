from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load libraries
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Keep dialog in memory 
chat_history = None

class Message(BaseModel):
    text: str

# Get sentiment status (between 1 star and 5 stars)
def get_sentiment(label: str) -> str:
    if label.startswith("1") or label.startswith("2"):
        return "Négatif"
    elif label.startswith("3"):
        return "Neutre"
    else:
        return "Positif"

# Api to translate and response to the given message
@app.post("/process/")
def process(msg: Message):
    global chat_history
    user_text = msg.text

    # Translation
    translation = translator(user_text)[0]["translation_text"]

    # Analyse sentiment
    sentiment_label = sentiment_analyzer(user_text)[0]["label"]
    sentiment = get_sentiment(sentiment_label)

    # Chatbot response
    input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")
    input_combined = torch.cat([chat_history, input_ids], dim=-1) if chat_history is not None else input_ids
    chat_history = model.generate(input_combined, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    bot_reply = tokenizer.decode(chat_history[:, input_combined.shape[-1]:][0], skip_special_tokens=True)

    return {
        "original": user_text,
        "translation": translation,
        "sentiment": sentiment,
        "response": bot_reply,
    }