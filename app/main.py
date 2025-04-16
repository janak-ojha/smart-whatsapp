# app/main.py
from fastapi import FastAPI
import logging
from . import twilio_webhook  # Import the router
from .database import engine, get_db  # Ensure your database is set up
from . import models  # Import your models to create tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Bot with Gemini API")

# Create database tables on startup
models.Base.metadata.create_all(bind=engine)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "message": "WhatsApp Bot is running"}

# Include the Twilio webhook router
app.include_router(twilio_webhook.router)