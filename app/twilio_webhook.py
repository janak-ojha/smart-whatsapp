from fastapi import APIRouter, Request, BackgroundTasks, Depends
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from sqlalchemy.orm import Session
from .database import get_db
from .message_processor import process_whatsapp_message

router = APIRouter()

@router.post("/twilio-webhook/")
async def twilio_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    form = await request.form()
    incoming_message = form.get('Body', '').strip()
    customer_number = form.get('From', '')
    print("customer_number:", customer_number)
    
    # Launch processing in background
    background_tasks.add_task(process_whatsapp_message, customer_number, incoming_message, db)
    twiml_response = MessagingResponse()
    # Either return an empty response or a very short "thinking" message
    # twiml_response.message("Processing your request...")
    
    return Response(content=str(twiml_response), media_type="application/xml")