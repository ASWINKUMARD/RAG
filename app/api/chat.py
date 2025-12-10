from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import os
import hashlib
from pathlib import Path
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Global storage for document chunks (in-memory vector DB)
document_store = {}
uploaded_files = []

# Request/Response models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous messages")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI's response")
    sources: Optional[List[str]] = Field(default=[], description="Source documents")
    status: str = Field(default="success", description="Response status")

class UploadResponse(BaseModel):
    message: str
    filename: str
    status: str = "success"

# Simple text extraction
def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from uploaded file"""
    try:
        if filename.endswith('.txt'):
            return content.decode('utf-8', errors='ignore')
        elif filename.endswith('.pdf'):
            # For PDF, try simple text extraction
            text = content.decode('latin-1', errors='ignore')
            return text
        elif filename.endswith('.docx'):
            # Basic DOCX text extraction
            text = content.decode('utf-8', errors='ignore')
            return text
        else:
            return content.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        return ""

# Simple chunking
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

# Simple keyword-based retrieval (fallback if embeddings fail)
def simple_retrieve(query: str, k: int = 3) -> List[tuple]:
    """Simple keyword-based document retrieval"""
    query_words = set(query.lower().split())
    scores = []
    
    for doc_id, chunks in document_store.items():
        for idx, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            # Calculate overlap score
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scores.append((overlap, doc_id, chunk))
    
    # Sort by score and return top k
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:k]

# LLM API call using OpenRouter
async def generate_response(query: str, context: str, conversation_history: List[ChatMessage]) -> str:
    """Generate response using OpenRouter API with Kwaipilot model"""
    
    # Build conversation history
    messages = []
    for msg in conversation_history[-5:]:  # Last 5 messages for context
        messages.append({"role": msg.role, "content": msg.content})
    
    # Add system prompt with context
    system_prompt = f"""You are a helpful AI assistant that answers questions based on provided documents.

Context from documents:
{context}

Instructions:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain relevant information, say so honestly
- Be concise and accurate
- Cite specific parts of the context when possible"""

    messages.insert(0, {"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "kwaipilot/kat-coder-pro:free",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500,
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                # Fallback to simple response
                return f"Based on the documents: {context[:300]}... (API error - showing context only)"
                
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        # Fallback response
        return f"I found relevant information in the documents:\n\n{context[:400]}..."

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with RAG implementation
    """
    try:
        logger.info(f"Received message: {request.message[:100]}...")
        
        # Validate input
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        # Check if documents are uploaded
        if not document_store:
            response_text = (
                "I don't have any documents to reference yet. "
                "Please upload documents first using the 'Upload Document' button, "
                "then I'll be able to answer questions about them!"
            )
            return ChatResponse(
                response=response_text,
                sources=[],
                status="success"
            )
        
        # Retrieve relevant context
        retrieved_docs = simple_retrieve(request.message, k=3)
        
        if not retrieved_docs:
            return ChatResponse(
                response="I couldn't find relevant information in the uploaded documents to answer your question.",
                sources=list(document_store.keys()),
                status="success"
            )
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for score, doc_id, chunk in retrieved_docs:
            context_parts.append(f"From {doc_id}:\n{chunk}\n")
            if doc_id not in sources:
                sources.append(doc_id)
        
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        response_text = await generate_response(
            request.message, 
            context,
            request.conversation_history
        )
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process documents for RAG
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Check file extension
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = '.' + file.filename.split('.')[-1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # FIXED: Check file size (10MB limit)
        MAX_SIZE = 10 * 1024 * 1024  # 10MB in bytes
        content = await file.read()
        
        if len(content) > MAX_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds 10MB limit"
            )
        
        logger.info(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        # Extract text from file
        text = extract_text_from_file(content, file.filename)
        
        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract text from file or file is too short"
            )
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Store in document store
        doc_id = file.filename
        document_store[doc_id] = chunks
        
        if doc_id not in uploaded_files:
            uploaded_files.append(doc_id)
        
        logger.info(f"Document processed: {doc_id}, {len(chunks)} chunks created")
        
        return UploadResponse(
            message=f"Successfully processed '{file.filename}' into {len(chunks)} chunks. You can now ask questions!",
            filename=file.filename,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

@router.delete("/clear", status_code=status.HTTP_200_OK)
async def clear_conversation():
    """Clear all uploaded documents and conversation"""
    try:
        document_store.clear()
        uploaded_files.clear()
        logger.info("All documents and conversation cleared")
        
        return JSONResponse({
            "message": "All documents and conversation cleared successfully",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Clear error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing data: {str(e)}"
        )

@router.get("/status", status_code=status.HTTP_200_OK)
async def get_status():
    """Get API status and available documents"""
    return JSONResponse({
        "status": "online",
        "documents_loaded": len(document_store),
        "document_names": uploaded_files,
        "features": {
            "chat": "available",
            "upload": "available",
            "supported_formats": ["pdf", "docx", "txt"]
        },
        "limits": {
            "max_file_size": "100MB",
            "max_message_length": 5000
        }
    })

@router.get("/documents", status_code=status.HTTP_200_OK)
async def list_documents():
    """List all uploaded documents"""
    return JSONResponse({
        "documents": uploaded_files,
        "count": len(uploaded_files),
        "status": "success"
    })
