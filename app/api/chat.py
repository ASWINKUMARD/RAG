from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import os
import re
from pathlib import Path
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Global storage for document chunks with metadata
document_store: Dict[str, List[Dict]] = {}
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
    chunks_created: int = 0
    status: str = "success"

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}\'\"]+', ' ', text)
    return text.strip()

def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from uploaded file with better handling"""
    try:
        if filename.endswith('.txt'):
            return content.decode('utf-8', errors='ignore')
        
        elif filename.endswith('.pdf'):
            # Try to extract text from PDF
            # Basic extraction - for production, use PyPDF2 or pdfplumber
            text = content.decode('latin-1', errors='ignore')
            # Clean up common PDF artifacts
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
            return clean_text(text)
        
        elif filename.endswith('.docx'):
            # Basic DOCX extraction - for production, use python-docx
            text = content.decode('utf-8', errors='ignore')
            return clean_text(text)
        
        else:
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Text extraction error for {filename}: {e}")
        return ""

def create_smart_chunks(text: str, chunk_size: int = 400, overlap: int = 100) -> List[Dict]:
    """
    Create smart overlapping chunks with metadata
    Returns list of dicts with 'text' and 'metadata'
    """
    chunks = []
    
    # Split by paragraphs first (better context preservation)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    current_chunk = []
    current_size = 0
    chunk_index = 0
    
    for para in paragraphs:
        words = para.split()
        para_size = len(words)
        
        # If paragraph alone exceeds chunk size, split it
        if para_size > chunk_size:
            if current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'index': chunk_index,
                    'words': set(chunk_text.lower().split()),
                    'size': len(current_chunk)
                })
                chunk_index += 1
                current_chunk = []
                current_size = 0
            
            # Split large paragraph
            for i in range(0, len(words), chunk_size - overlap):
                para_chunk = ' '.join(words[i:i + chunk_size])
                chunks.append({
                    'text': para_chunk,
                    'index': chunk_index,
                    'words': set(para_chunk.lower().split()),
                    'size': chunk_size
                })
                chunk_index += 1
        
        # If adding paragraph exceeds chunk size, save current and start new
        elif current_size + para_size > chunk_size:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'index': chunk_index,
                    'words': set(chunk_text.lower().split()),
                    'size': current_size
                })
                chunk_index += 1
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                overlap_words = ' '.join(current_chunk[-overlap:]).split()
                current_chunk = overlap_words + words
                current_size = len(current_chunk)
            else:
                current_chunk = words
                current_size = para_size
        
        # Add paragraph to current chunk
        else:
            current_chunk.extend(words)
            current_size += para_size
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'index': chunk_index,
            'words': set(chunk_text.lower().split()),
            'size': current_size
        })
    
    return chunks

def advanced_retrieve(query: str, k: int = 4) -> List[tuple]:
    """
    Advanced retrieval with better scoring:
    - Exact phrase matching
    - Synonym handling
    - TF-IDF-like scoring
    """
    # Normalize query
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Remove common stop words for better matching
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for'}
    query_keywords = query_words - stop_words
    
    if not query_keywords:
        query_keywords = query_words
    
    scores = []
    
    for doc_id, chunks in document_store.items():
        for chunk_data in chunks:
            chunk_text = chunk_data['text']
            chunk_lower = chunk_text.lower()
            chunk_words = chunk_data['words']
            
            score = 0
            
            # 1. Exact phrase match (highest priority)
            if query_lower in chunk_lower:
                score += 100
            
            # 2. All query words present
            if query_keywords.issubset(chunk_words):
                score += 50
            
            # 3. Keyword overlap
            overlap = len(query_keywords & chunk_words)
            if overlap > 0:
                # Weight by percentage of query keywords found
                score += (overlap / len(query_keywords)) * 30
            
            # 4. Individual keyword matches with context
            for keyword in query_keywords:
                if keyword in chunk_lower:
                    # Bonus for keyword appearing multiple times
                    count = chunk_lower.count(keyword)
                    score += count * 10
                    
                    # Bonus for keyword in important positions (start/end)
                    words_list = chunk_lower.split()
                    if words_list and (words_list[0] == keyword or words_list[-1] == keyword):
                        score += 5
            
            # 5. Length penalty (prefer concise relevant chunks)
            if score > 0:
                length_penalty = min(1.0, 500 / len(chunk_text.split()))
                score *= length_penalty
            
            if score > 0:
                scores.append((score, doc_id, chunk_text, chunk_data['index']))
    
    # Sort by score (descending)
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # Log top matches for debugging
    logger.info(f"Query: '{query}' - Found {len(scores)} matching chunks")
    if scores:
        logger.info(f"Top score: {scores[0][0]:.2f}")
    
    return scores[:k]

async def generate_response(query: str, context: str, conversation_history: List[ChatMessage]) -> str:
    """Generate response using OpenRouter API"""
    
    # Build conversation
    messages = []
    for msg in conversation_history[-3:]:  # Last 3 for context
        messages.append({"role": msg.role, "content": msg.content})
    
    # System prompt
    system_prompt = f"""You are a helpful assistant that answers questions based on provided document context.

CONTEXT FROM DOCUMENTS:
{context}

INSTRUCTIONS:
- Answer the question using ONLY information from the context above
- Be specific and detailed when the context provides information
- If the context contains relevant information, extract and explain it clearly
- If the context doesn't answer the question, say so honestly
- Always cite which part of the document your answer comes from
- Use natural, conversational language"""

    messages.insert(0, {"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    
    try:
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        if not api_key:
            logger.warning("No OpenRouter API key found")
            return f"I found this relevant information in the documents:\n\n{context[:500]}...\n\n(Note: LLM API key not configured, showing raw context)"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "kwaipilot/kat-coder-pro:free", 
                    "messages": messages,
                    "temperature": 0.3, 
                    "max_tokens": 600,
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"LLM API error: {response.status_code}")
                # Fallback to context
                return f"Based on the documents:\n\n{context[:600]}...\n\n(API error - showing relevant excerpts)"
                
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return f"Here's what I found in the documents:\n\n{context[:600]}..."

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    """Main chat endpoint with improved RAG"""
    try:
        logger.info(f"Chat request: '{request.message[:100]}'")
        
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        # Check for documents
        if not document_store:
            return ChatResponse(
                response="I don't have any documents loaded yet. Please upload documents first using the 'Upload Document' button, then I can answer questions about them!",
                sources=[],
                status="success"
            )
        
        # Retrieve relevant chunks
        retrieved = advanced_retrieve(request.message, k=4)
        
        if not retrieved:
            return ChatResponse(
                response=f"I couldn't find information about '{request.message}' in the uploaded documents. Try rephrasing your question or asking about different topics covered in the documents.",
                sources=list(document_store.keys()),
                status="success"
            )
        
        # Build context and track sources
        context_parts = []
        sources = []
        seen_docs = set()
        
        for score, doc_id, chunk_text, chunk_idx in retrieved:
            context_parts.append(f"[From {doc_id}, Section {chunk_idx+1}]:\n{chunk_text}")
            if doc_id not in seen_docs:
                sources.append(doc_id)
                seen_docs.add(doc_id)
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Retrieved {len(retrieved)} chunks from {len(sources)} documents")
        
        # Generate response
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
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents"""
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Validate extension
        allowed = ['.pdf', '.docx', '.txt']
        ext = '.' + file.filename.split('.')[-1].lower()
        
        if ext not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed)}"
            )
        
        # Check size (10MB)
        content = await file.read()
        MAX_SIZE = 100 * 1024 * 1024
        
        if len(content) > MAX_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File exceeds 100MB limit"
            )
        
        logger.info(f"Processing: {file.filename} ({len(content)} bytes)")
        
        # Extract text
        text = extract_text_from_file(content, file.filename)
        
        if not text or len(text.strip()) < 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract sufficient text from file"
            )
        
        logger.info(f"Extracted {len(text)} characters from {file.filename}")
        
        # Create smart chunks
        chunks = create_smart_chunks(text, chunk_size=400, overlap=100)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create document chunks"
            )
        
        # Store chunks
        doc_id = file.filename
        document_store[doc_id] = chunks
        
        if doc_id not in uploaded_files:
            uploaded_files.append(doc_id)
        
        logger.info(f"✓ Stored {len(chunks)} chunks for {doc_id}")
        
        # Log sample for debugging
        if chunks:
            logger.info(f"Sample chunk: {chunks[0]['text'][:100]}...")
        
        return UploadResponse(
            message=f"Successfully processed '{file.filename}'",
            filename=file.filename,
            chunks_created=len(chunks),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )

@router.delete("/clear")
async def clear_conversation():
    """Clear all data"""
    try:
        document_store.clear()
        uploaded_files.clear()
        logger.info("Cleared all documents")
        
        return JSONResponse({
            "message": "All documents cleared",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/status")
async def get_status():
    """API status"""
    total_chunks = sum(len(chunks) for chunks in document_store.values())
    
    return JSONResponse({
        "status": "online",
        "documents_loaded": len(document_store),
        "total_chunks": total_chunks,
        "document_names": uploaded_files,
        "features": {
            "chat": "available",
            "upload": "available",
            "advanced_retrieval": "enabled",
            "supported_formats": ["pdf", "docx", "txt"]
        },
        "limits": {
            "max_file_size": "100MB",
            "chunk_size": 800,
            "chunk_overlap": 200
        }
    })

@router.get("/documents")
async def list_documents():
    """List documents with details"""
    docs_info = []
    
    for doc_id in uploaded_files:
        if doc_id in document_store:
            chunks = document_store[doc_id]
            total_words = sum(chunk['size'] for chunk in chunks)
            docs_info.append({
                "filename": doc_id,
                "chunks": len(chunks),
                "total_words": total_words
            })
    
    return JSONResponse({
        "documents": docs_info,
        "count": len(uploaded_files),
        "status": "success"
    })
