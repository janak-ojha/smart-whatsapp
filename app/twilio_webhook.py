from fastapi import APIRouter, Request, BackgroundTasks, Depends,UploadFile,File,HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from sqlalchemy.orm import Session
from .database import get_db
from .message_processor import process_whatsapp_message
from .schemas import DeviceDocumentResponse
import logging
import os
from .models import DeviceDocument
from datetime import datetime
import uuid
from fastapi import Form
import shutil
from .document_preprocessing import process_and_store_embeddings


router = APIRouter()
logger = logging.getLogger(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@router.post("/upload-device-document/", response_model=DeviceDocumentResponse)
async def upload_device_document(
    device_id: int = Form(...),
    device_pdf: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not device_pdf.filename.endswith(".pdf"):
        logger.warning("Upload failed: Non-PDF file attempted")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # Save PDF to disk
        unique_filename = f"{uuid.uuid4()}_{device_pdf.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(device_pdf.file, buffer)
        logger.info(f"File saved to disk: {file_path}")

        # Save metadata to DB
        new_doc = DeviceDocument(
            device_id=device_id,
            device_pdf=unique_filename,
            timestamp=datetime.utcnow()
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        logger.info(f"Document saved in DB: id={new_doc.id}, device_id={device_id}, filename={unique_filename}")

        # Process embeddings and store in Pinecone with the device_id
        logger.info(f"Starting embedding process for file: {file_path}, device_id: {device_id}")
        process_and_store_embeddings(file_path, device_id)
        logger.info(f"Embedding process completed for device_id: {device_id}")

        # Optional: delete the file after embedding
        os.remove(file_path)
        logger.info(f"File deleted after processing: {file_path}")

        return new_doc

    except Exception as e:
        logger.error(f"Upload or embedding failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving and processing document: {str(e)}")
