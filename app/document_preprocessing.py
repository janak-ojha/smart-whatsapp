import os
import logging
import requests
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import time
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone with error handling and index connection
def initialize_pinecone():
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not pinecone_api_key or not index_name:
            raise ValueError("Missing Pinecone API key or index name in environment variables")
        
        # Initialize Pinecone client using Pinecone class
        pc = Pinecone(api_key=pinecone_api_key)
        
        # List existing indexes
        existing_indexes = pc.list_indexes().names()
        
        # Check if the index exists, if not, create it
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=768,  # Must match your embedding model's output
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            logger.info(f"Created new Pinecone index: {index_name}")
            # Wait for index to be ready (you can adjust this depending on your needs)
            time.sleep(5)  # Wait 1 minute for index initialization
        else:
            logger.info(f"Using existing Pinecone index: {index_name}")
        
        # Connect to the index
        index = pc.Index(index_name)
        logger.info("Successfully connected to Pinecone index")
        return index
        
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

# Enhanced document loading and chunking with parallel processing for large PDFs
def load_and_chunk_pdf(pdf_path: str):
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        logger.info(f"Loading PDF: {pdf_path}")

        # First try direct fitz approach as a fallback
        try:
            doc = fitz.open(pdf_path)
            logger.info(f"Successfully opened PDF with fitz directly: {pdf_path}")
            doc.close()
        except Exception as e:
            logger.warning(f"Direct fitz test failed: {str(e)}")

        # Then try the loader approach
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            logger.info("Successfully loaded documents with PyMuPDFLoader")
        except ImportError:
            logger.error("PyMuPDFLoader failed - trying manual loading with fitz")
            # Fallback implementation if PyMuPDFLoader fails
            from langchain_core.documents import Document
            documents = []
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                # Create Document objects instead of dictionaries
                documents.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "page": page_num}
                ))
            doc.close()

        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split PDF into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}") 
        raise

# Optimized embedding retrieval with retries and better error handling
def get_embeddings(texts: list, max_retries: int = 3):
    HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    if not HF_API_KEY:
        raise ValueError("Missing HuggingFace API key in environment variables")
    
    HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Attempt {attempt + 1}: API returned {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}: Request failed - {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)
    
    raise Exception(f"Failed to get embeddings after {max_retries} attempts")

# Embedding processing and storage using parallelism for faster embedding handling
def process_and_store_embeddings(pdf_path: str, device_id: int, batch_size: int = 64):
    try:
        logger.info(f"Device ID: {device_id}")
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Initialize Pinecone index
        index = initialize_pinecone()
        
        # Load and chunk PDF
        chunks = load_and_chunk_pdf(pdf_path)
        logger.info(f"Total Chunks: {len(chunks)}")

        # Using ThreadPoolExecutor to parallelize the embedding retrieval process
        with ThreadPoolExecutor(max_workers=5) as executor:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk.page_content for chunk in batch]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

                # Retrieve embeddings concurrently
                embeddings = executor.submit(get_embeddings, texts)
                embeddings = embeddings.result()

                # Prepare upsert data for Pinecone
                vectors = []
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    metadata = {
                        "device_id": device_id,
                        "chunk_num": i + j
                    }
                    vectors.append({
                        "id": f"device-{device_id}-chunk-{i + j}",
                        "values": embedding,
                        "metadata": metadata
                    })
                
                # Upsert to Pinecone in batches
                index.upsert(vectors=vectors)
                logger.info(f"Uploaded batch {i//batch_size + 1} to Pinecone")

    except Exception as e:
        logger.error(f"Error in processing pipeline: {str(e)}")
        raise
