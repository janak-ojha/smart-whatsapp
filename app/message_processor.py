import logging
import os
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi.concurrency import run_in_threadpool
from twilio.rest import Client
from .models import User, Message
from .gemini import GeminiLLMClient  # Import the Gemini LLM client

logger = logging.getLogger(__name__)

# Initialize Twilio client
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')  # Your WhatsApp Business number with whatsapp: prefix

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# WhatsApp message character limit
WHATSAPP_MAX_LENGTH = 1500  # Using 1500 instead of 1600 for safety

async def process_whatsapp_message(sender: str, message_text: str, db: Session) -> str:
    """
    Processes an incoming WhatsApp message:
      - Finds or creates a user by phone number.
      - Saves the incoming message.
      - Uses GeminiLLMClient to get a plain-text response.
      - Saves the outgoing message.
      - Sends the response back via WhatsApp, handling long messages.
    
    Returns the Gemini API response text.
    """
    try:
        logger.info(f"Processing message from {sender}: {message_text}")

        # Get or create the user by phone number.
        user = db.query(User).filter(User.phone == sender).first()
        if not user:
            user = User(phone=sender)
            db.add(user)
            db.commit()
            db.refresh(user)

        # Save the incoming message.
        incoming_message = Message(
            user_id=user.id,
            content=message_text,
            direction="incoming"
        )
        db.add(incoming_message)
        db.commit()
        db.refresh(incoming_message)

        # Optionally, retrieve chat history for context
        # This could be expanded to get the last few messages
        chat_history = []

        # Get response from GeminiLLMClient (wrapped in threadpool to avoid blocking)
        gemini_client = GeminiLLMClient()
        response_text = await run_in_threadpool(
            gemini_client.generate_response, 
            message_text, 
            chat_history,
            max_length=700  # Limit Gemini response length to avoid WhatsApp issues
        )

        # Save the outgoing message.
        outgoing_message = Message(
            user_id=user.id,
            content=response_text,
            direction="outgoing"
        )
        db.add(outgoing_message)
        db.commit()
        db.refresh(outgoing_message)

        # Send the response back via WhatsApp, handling long messages
        await run_in_threadpool(
            send_whatsapp_message, 
            to_number=sender,
            message_body=response_text
        )

        logger.info(f"Successfully processed message from {sender} and sent response")
        return response_text

    except SQLAlchemyError as db_err:
        logger.error(f"Database error: {str(db_err)}")
        db.rollback()
        error_msg = "Sorry, there was a problem processing your message."
        # Try to send error message to user
        await run_in_threadpool(send_whatsapp_message, sender, error_msg)
        return error_msg
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        error_msg = "Sorry, an unexpected error occurred."
        # Try to send error message to user
        await run_in_threadpool(send_whatsapp_message, sender, error_msg)
        return error_msg

def send_whatsapp_message(to_number: str, message_body: str):
    """
    Sends a WhatsApp message using the Twilio API.
    Handles long messages by splitting them into multiple messages if needed.
    """
    try:
        # Clean up the recipient number (remove "whatsapp:" if present)
        cleaned_to = to_number.replace("whatsapp:", "")
        # Add WhatsApp prefix properly
        formatted_to = f"whatsapp:{cleaned_to}"
        
        if TWILIO_PHONE_NUMBER == cleaned_to or TWILIO_PHONE_NUMBER == formatted_to:
            logger.error(f"Cannot send: Twilio number {TWILIO_PHONE_NUMBER} matches recipient {to_number}")
            raise ValueError("Sender and recipient cannot be the same number")
        
        formatted_from = TWILIO_PHONE_NUMBER
        if not formatted_from.startswith("whatsapp:"):
            formatted_from = f"whatsapp:{formatted_from}"
        
        logger.info(f"Sending from {formatted_from} to {formatted_to}")
        
        # Check if message needs to be split
        if len(message_body) <= WHATSAPP_MAX_LENGTH:
            # Send as a single message
            message = twilio_client.messages.create(
                body=message_body,
                from_=formatted_from,
                to=formatted_to
            )
            logger.info(f"Sent WhatsApp message: SID={message.sid}")
            return message.sid
        else:
            # Split message into multiple parts
            message_parts = split_long_message(message_body)
            message_sids = []
            
            for i, part in enumerate(message_parts):
                prefix = f"Message part {i+1}/{len(message_parts)}: " if len(message_parts) > 1 else ""
                message = twilio_client.messages.create(
                    body=f"{prefix}{part}",
                    from_=formatted_from,
                    to=formatted_to
                )
                message_sids.append(message.sid)
                logger.info(f"Sent WhatsApp message part {i+1}/{len(message_parts)}: SID={message.sid}")
            
            return message_sids
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {str(e)}")
        raise

def split_long_message(message: str) -> list:
    """
    Splits a long message into multiple parts, each under WHATSAPP_MAX_LENGTH characters.
    Tries to split at paragraph or sentence boundaries when possible.
    """
    if len(message) <= WHATSAPP_MAX_LENGTH:
        return [message]
    
    parts = []
    remaining = message
    
    while len(remaining) > WHATSAPP_MAX_LENGTH:
        # Try to find a good split point (paragraph)
        split_index = remaining[:WHATSAPP_MAX_LENGTH].rfind('\n\n')
        
        if split_index == -1 or split_index < WHATSAPP_MAX_LENGTH // 2:
            # No paragraph break, try sentence break
            split_index = remaining[:WHATSAPP_MAX_LENGTH].rfind('. ')
            if split_index != -1:
                split_index += 1  # Include the period
        
        if split_index == -1 or split_index < WHATSAPP_MAX_LENGTH // 2:
            # No sentence break, try any space
            split_index = remaining[:WHATSAPP_MAX_LENGTH].rfind(' ')
        
        if split_index == -1 or split_index < WHATSAPP_MAX_LENGTH // 2:
            # No good break point, just split at the max length
            split_index = WHATSAPP_MAX_LENGTH - 1
        
        parts.append(remaining[:split_index].strip())
        remaining = remaining[split_index:].strip()
    
    if remaining:
        parts.append(remaining)
    
    return parts