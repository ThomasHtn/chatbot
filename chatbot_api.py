from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from datetime import datetime
import os
import torch

app = FastAPI()

# Add "logs" folder if doesn't exist
os.makedirs("logs", exist_ok=True)

# Create log file based on current date
log_file = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
logger.add(log_file, rotation="00:00", retention="7 days", level="INFO")

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
    logger.info(f"User message : {user_text}")

    try:
        # Translation
        translation = translator(user_text)[0]["translation_text"]
        logger.info(f"Translation EN : {translation}")

        # Analyse sentiment
        sentiment_label = sentiment_analyzer(user_text)[0]["label"]
        sentiment = get_sentiment(sentiment_label)
        logger.info(f"Sentiment : {sentiment} ({sentiment_label})")

        # Chatbot response
        input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")
        input_combined = torch.cat([chat_history, input_ids], dim=-1) if chat_history is not None else input_ids
        chat_history = model.generate(input_combined, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        bot_reply = tokenizer.decode(chat_history[:, input_combined.shape[-1]:][0], skip_special_tokens=True)
        logger.info(f"Chatbot response : {bot_reply}")
        
        result = {
            "original": user_text,
            "translation": translation,
            "sentiment": sentiment,
            "response": bot_reply,
        }

        logger.success(f"Response : {result}")

        return result
    
    except Exception as e:
            logger.error(f"An error occured: {e}")
            raise HTTPException(status_code=500, detail=str(e))