import os
import logging
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()

# --------- Embedding Client ---------
class HuggingFaceClient:
    def __init__(self):
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        return response.json()

# --------- Gemini LLM Client ---------
class GeminiLLMClients:
    def __init__(self, api_key: str):
        self.api_key = api_key  # use the argument passed
        self.model = "gemini-1.5-flash"  # or "gemini-pro", "gemini-1.5-pro", etc.
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.headers = {"Content-Type": "application/json"}

    def generate_response(self, query: str, context: str = "") -> str:
        prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 1.0,
                "maxOutputTokens": 512
            }
        }
        try:
            full_url = f"{self.url}?key={self.api_key}"
            response = requests.post(full_url, headers=self.headers, json=payload)
            response.raise_for_status()
            content = response.json()
            return content["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Sorry, something went wrong while generating the response."

# --------- Pinecone Initialization ---------
def initialize_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)

# --------- Retrieval Function ---------
def retrieve_context_from_pinecone(prompt: str, device_id: int, index, embedding_client, top_k: int = 5) -> str:
    try:
        query_embedding = embedding_client.get_embeddings([prompt])[0]
        response = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"device_id": {"$eq": device_id}},
        )
        matches = response.get("matches", [])
        if not matches:
            logger.warning("No context matches found.")
            return ""
        context = "\n\n".join([match["metadata"]["text"] for match in matches])
        logger.info(f"Retrieved {len(matches)} context chunks.")
        return context
    except Exception as e:
        logger.error(f"Error during context retrieval: {e}")
        return ""

# --------- RAG Execution Function ---------
def answer_question_with_rag(question: str, device_id: int):
    # Init clients
    embedding_client = HuggingFaceClient()
    pinecone_index = initialize_pinecone()
    gemini_client = GeminiLLMClients(api_key=os.getenv("GEMINI_API_KEY"))

    # Retrieve context
    context = retrieve_context_from_pinecone(
        prompt=question,
        device_id=device_id,
        index=pinecone_index,
        embedding_client=embedding_client
    )

    # Generate final answer
    return gemini_client.generate_response(query=question, context=context)


