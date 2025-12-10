from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import os
import re
from pathlib import Path
import httpx
import io

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
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}\'\"]+', ' ', text)
    return text.strip()

def extract_text_from_pdf_improved(content: bytes) -> str:
    """
    Improved PDF text extraction using PyPDF2
    Falls back to basic extraction if library not available
    """
    try:
        # Try using PyPDF2 first (best method)
        try:
            import PyPDF2
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            
            full_text = '\n'.join(text_parts)
            
            if full_text and len(full_text.strip()) > 50:
                logger.info(f"✓ Extracted {len(full_text)} chars using PyPDF2")
                return clean_text(full_text)
        except ImportError:
            logger.warning("PyPDF2 not installed, using fallback method")
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}, using fallback")
        
        # Fallback: Try pdfplumber
        try:
            import pdfplumber
            pdf_file = io.BytesIO(content)
            text_parts = []
            
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            full_text = '\n'.join(text_parts)
            
            if full_text and len(full_text.strip()) > 50:
                logger.info(f"✓ Extracted {len(full_text)} chars using pdfplumber")
                return clean_text(full_text)
        except ImportError:
            logger.warning("pdfplumber not installed")
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Last resort: Basic text extraction
        logger.warning("Using basic text extraction - may have poor quality")
        text = content.decode('latin-1', errors='ignore')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        return clean_text(text)
        
    except Exception as e:
        logger.error(f"All PDF extraction methods failed: {e}")
        return ""

def extract_text_from_docx_improved(content: bytes) -> str:
    """
    Improved DOCX text extraction using python-docx
    """
    try:
        # Try using python-docx
        try:
            import docx
            docx_file = io.BytesIO(content)
            doc = docx.Document(docx_file)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text_parts.append(cell.text)
            
            full_text = '\n'.join(text_parts)
            
            if full_text and len(full_text.strip()) > 20:
                logger.info(f"✓ Extracted {len(full_text)} chars using python-docx")
                return clean_text(full_text)
        except ImportError:
            logger.warning("python-docx not installed, using fallback")
        except Exception as e:
            logger.warning(f"python-docx extraction failed: {e}")
        
        # Fallback
        text = content.decode('utf-8', errors='ignore')
        return clean_text(text)
        
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""

def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from uploaded file with improved handling"""
    try:
        if filename.endswith('.txt'):
            return content.decode('utf-8', errors='ignore')
        
        elif filename.endswith('.pdf'):
            return extract_text_from_pdf_improved(content)
        
        elif filename.endswith('.docx'):
            return extract_text_from_docx_improved(content)
        
        else:
            # Try as plain text
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Text extraction error for {filename}: {e}")
        return ""

def create_smart_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Create smart overlapping chunks with better context preservation
    """
    chunks = []
    
    # Remove excessive whitespace but keep paragraph structure
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Split by double newlines (paragraphs) first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # If no clear paragraphs, split by single newlines
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    # If still no structure, split by sentences
    if len(paragraphs) <= 1:
        import re
        paragraphs = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = []
    current_size = 0
    chunk_index = 0
    
    for para in paragraphs:
        words = para.split()
        para_size = len(words)
        
        # If paragraph alone exceeds chunk size, split it
        if para_size > chunk_size:
            # Save current chunk first
            if current_chunk:
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
            
            # Split large paragraph into smaller chunks
            for i in range(0, len(words), chunk_size - overlap):
                para_chunk = ' '.join(words[i:i + chunk_size])
                chunks.append({
                    'text': para_chunk,
                    'index': chunk_index,
                    'words': set(para_chunk.lower().split()),
                    'size': min(chunk_size, len(words) - i)
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
            
            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                overlap_text = ' '.join(current_chunk[-overlap:])
                current_chunk = overlap_text.split() + words
                current_size = len(current_chunk)
            else:
                current_chunk = words
                current_size = para_size
        
        # Add paragraph to current chunk
        else:
            current_chunk.extend(words)
            current_size += para_size
    
    # Save the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'index': chunk_index,
            'words': set(chunk_text.lower().split()),
            'size': current_size
        })
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks

def advanced_retrieve(query: str, k: int = 5) -> List[tuple]:
    """
    Advanced retrieval with semantic matching
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Remove common stop words
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as'}
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
            
            # 1. Exact phrase match (very high priority)
            if query_lower in chunk_lower:
                score += 150
            
            # 2. All query keywords present
            if query_keywords.issubset(chunk_words):
                score += 80
            
            # 3. Partial keyword matches
            overlap = len(query_keywords & chunk_words)
            if overlap > 0:
                overlap_ratio = overlap / len(query_keywords)
                score += overlap_ratio * 50
            
            # 4. Individual keyword scoring
            for keyword in query_keywords:
                if keyword in chunk_lower:
                    # Count occurrences
                    count = chunk_lower.count(keyword)
                    score += count * 15
                    
                    # Bonus for keyword at start
                    if chunk_lower.startswith(keyword):
                        score += 10
            
            # 5. Proximity bonus (keywords close together)
            if len(query_keywords) > 1:
                words_list = chunk_lower.split()
                keyword_positions = []
                for i, word in enumerate(words_list):
                    if any(kw in word for kw in query_keywords):
                        keyword_positions.append(i)
                
                if len(keyword_positions) >= 2:
                    # Calculate average distance between keywords
                    distances = [keyword_positions[i+1] - keyword_positions[i] 
                               for i in range(len(keyword_positions)-1)]
                    avg_distance = sum(distances) / len(distances)
                    
                    # Closer keywords = higher score
                    if avg_distance < 10:
                        score += 30 / avg_distance
            
            # 6. Length normalization (prefer concise relevant chunks)
            if score > 0:
                chunk_length = len(chunk_text.split())
                length_penalty = min(1.0, 600 / chunk_length) if chunk_length > 0 else 0
                score *= length_penalty
            
            if score > 5:  # Only include chunks with meaningful scores
                scores.append((score, doc_id, chunk_text, chunk_data['index']))
    
    # Sort by score descending
    scores.sort(reverse=True, key=lambda x: x[0])
    
    logger.info(f"Query: '{query}' - Found {len(scores)} matching chunks")
    if scores:
        logger.info(f"Top 3 scores: {[f'{s[0]:.1f}' for s in scores[:3]]}")
    
    return scores[:k]

async def generate_response(query: str, context: str, conversation_history: List[ChatMessage]) -> str:
    """Generate response using OpenRouter API"""
    
    messages = []
    for msg in conversation_history[-3:]:
        messages.append({"role": msg.role, "content": msg.content})
    
    system_prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer the user's question using ONLY the information from the context above
- Be specific, detailed, and well-structured in your response
- If the context contains relevant information, extract and explain it clearly
- Use bullet points or numbered lists when appropriate for clarity
- If the context doesn't contain the answer, honestly say so
- Always indicate which document or section your answer comes from
- Use natural, conversational language"""

    messages.insert(0, {"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    
    try:
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        if not api_key:
            logger.warning("No OpenRouter API key - returning context")
            return f"Based on the documents:\n\n{context[:800]}\n\n(Note: Configure OPENROUTER_API_KEY for better AI responses)"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 800,
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return f"Based on the documents:\n\n{context[:800]}"
                
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return f"Here's what I found in the documents:\n\n{context[:800]}"

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"Chat request: '{request.message[:100]}'")
        
        if not request.message or not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty"
            )
        
        if not document_store:
            return ChatResponse(
                response="I don't have any documents loaded yet. Please upload documents first using the 'Upload Document' button, then I can answer questions about them!",
                sources=[],
                status="success"
            )
        
        # Retrieve relevant chunks
        retrieved = advanced_retrieve(request.message, k=5)
        
        if not retrieved:
            return ChatResponse(
                response=f"I couldn't find relevant information about '{request.message}' in the uploaded documents. The documents cover: {', '.join(document_store.keys())}. Try asking about topics mentioned in these documents.",
                sources=list(document_store.keys()),
                status="success"
            )
        
        # Build context
        context_parts = []
        sources = []
        seen_docs = set()
        
        for score, doc_id, chunk_text, chunk_idx in retrieved:
            context_parts.append(f"[Document: {doc_id}, Section {chunk_idx+1}, Relevance: {score:.1f}]\n{chunk_text}")
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
        
        allowed = ['.pdf', '.docx', '.txt']
        ext = '.' + file.filename.split('.')[-1].lower()
        
        if ext not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed)}"
            )
        
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
                detail="Could not extract sufficient text from file. The file may be empty, corrupted, or contain only images."
            )
        
        logger.info(f"Extracted {len(text)} characters from {file.filename}")
        logger.info(f"Sample text: {text[:200]}...")
        
        # Create chunks
        chunks = create_smart_chunks(text, chunk_size=500, overlap=100)
        
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
        
        return UploadResponse(
            message=f"Successfully processed '{file.filename}' - extracted {len(text)} characters into {len(chunks)} searchable chunks",
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
            "pdf_extraction": "PyPDF2/pdfplumber",
            "supported_formats": ["pdf", "docx", "txt"]
        }
    })

@router.get("/documents")
async def list_documents():
    """List documents"""
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
