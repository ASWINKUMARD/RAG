from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import os
import io
import httpx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Global storage
document_store: Dict[str, List[Dict]] = {}
uploaded_files = []

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_history: Optional[List[ChatMessage]] = Field(default=[])

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = Field(default=[])
    status: str = "success"

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int = 0
    status: str = "success"

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF using PyPDF2"""
    try:
        import PyPDF2
        
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                continue
        
        full_text = '\n\n'.join(text_parts)
        
        if full_text and len(full_text.strip()) > 50:
            logger.info(f"✓ Extracted {len(full_text)} chars from PDF")
            return full_text
        else:
            logger.error("PDF extraction resulted in insufficient text")
            return ""
            
    except ImportError:
        logger.error("PyPDF2 not installed!")
        raise HTTPException(
            status_code=500,
            detail="PyPDF2 library not installed. Add 'PyPDF2==3.0.1' to requirements.txt"
        )
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX"""
    try:
        import docx
        
        docx_file = io.BytesIO(content)
        doc = docx.Document(docx_file)
        
        text_parts = []
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)
        
        full_text = '\n\n'.join(text_parts)
        
        if full_text and len(full_text.strip()) > 20:
            logger.info(f"✓ Extracted {len(full_text)} chars from DOCX")
            return full_text
        else:
            return ""
            
    except ImportError:
        logger.error("python-docx not installed!")
        raise HTTPException(
            status_code=500,
            detail="python-docx library not installed. Add 'python-docx==1.1.0' to requirements.txt"
        )
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""

def extract_text_from_file(content: bytes, filename: str) -> str:
    """Main text extraction function"""
    try:
        if filename.lower().endswith('.txt'):
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text = content.decode(encoding)
                    if text and len(text.strip()) > 20:
                        logger.info(f"✓ Decoded TXT with {encoding}")
                        return text
                except:
                    continue
            return ""
        
        elif filename.lower().endswith('.pdf'):
            return extract_text_from_pdf(content)
        
        elif filename.lower().endswith('.docx'):
            return extract_text_from_docx(content)
        
        else:
            # Try as plain text
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Text extraction error: {e}")
        return ""

def create_chunks(text: str, chunk_size: int = 600, overlap: int = 150) -> List[Dict]:
    """Create text chunks with overlap"""
    
    # Clean text
    import re
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    if not text:
        return []
    
    # Split into sentences (approximate)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        # If adding this sentence exceeds chunk size
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'index': chunk_idx,
                'words': set(chunk_text.lower().split())
            })
            chunk_idx += 1
            
            # Start new chunk with overlap
            if overlap > 0:
                # Keep last few words for overlap
                overlap_words = current_chunk[-overlap:]
                current_chunk = overlap_words + sentence_words
                current_length = len(current_chunk)
            else:
                current_chunk = sentence_words
                current_length = sentence_length
        else:
            current_chunk.extend(sentence_words)
            current_length += sentence_length
    
    # Save last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'index': chunk_idx,
            'words': set(chunk_text.lower().split())
        })
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[tuple]:
    """Retrieve most relevant chunks using keyword matching"""
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Remove stop words
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of'}
    query_keywords = query_words - stop_words
    
    if not query_keywords:
        query_keywords = query_words
    
    results = []
    
    for doc_id, chunks in document_store.items():
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_lower = chunk_text.lower()
            chunk_words = chunk['words']
            
            score = 0
            
            # Exact phrase match
            if query_lower in chunk_lower:
                score += 200
            
            # All keywords present
            if query_keywords.issubset(chunk_words):
                score += 100
            
            # Partial keyword match
            overlap = len(query_keywords & chunk_words)
            if overlap > 0:
                score += (overlap / len(query_keywords)) * 50
            
            # Individual keyword scoring
            for keyword in query_keywords:
                count = chunk_lower.count(keyword)
                score += count * 20
            
            if score > 0:
                results.append((score, doc_id, chunk_text, chunk['index']))
    
    # Sort by score
    results.sort(reverse=True, key=lambda x: x[0])
    
    logger.info(f"Found {len(results)} relevant chunks for: {query}")
    if results:
        logger.info(f"Top score: {results[0][0]:.1f}")
    
    return results[:top_k]

async def generate_llm_response(query: str, context: str) -> str:
    """Generate response using OpenRouter API"""
    
    system_prompt = f"""You are a helpful assistant that answers questions based on provided documents.

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer the user's question using ONLY the information from the context above
- Be clear, specific, and well-organized
- If the context contains the answer, provide it with details
- If the context doesn't contain the answer, say so honestly
- Use bullet points or lists when appropriate
- Cite which document section your answer comes from"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        if not api_key:
            logger.warning("No OPENROUTER_API_KEY found")
            return f"Based on the documents:\n\n{context[:1000]}\n\n(Configure OPENROUTER_API_KEY for AI-generated responses)"
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://rag-chatbot.com",
                    "X-Title": "RAG Chatbot"
                },
                json={
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return f"Based on the documents:\n\n{context[:1000]}"
                
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return f"Based on the documents:\n\n{context[:1000]}"

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    try:
        logger.info(f"Query: {request.message}")
        
        if not document_store:
            return ChatResponse(
                response="Please upload documents first! Click the 'Upload Document' button above to get started.",
                sources=[],
                status="success"
            )
        
        # Retrieve relevant chunks
        retrieved = retrieve_relevant_chunks(request.message, top_k=5)
        
        if not retrieved:
            return ChatResponse(
                response=f"I couldn't find relevant information about '{request.message}' in the uploaded documents. Try asking about: {', '.join(list(document_store.keys())[:3])}",
                sources=list(document_store.keys()),
                status="success"
            )
        
        # Build context
        context_parts = []
        sources = set()
        
        for score, doc_id, chunk_text, chunk_idx in retrieved:
            context_parts.append(f"[{doc_id} - Section {chunk_idx + 1}]\n{chunk_text}")
            sources.add(doc_id)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate response
        response_text = await generate_llm_response(request.message, context)
        
        return ChatResponse(
            response=response_text,
            sources=list(sources),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload document endpoint"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed = ['.pdf', '.docx', '.txt']
        ext = '.' + file.filename.rsplit('.', 1)[-1].lower()
        
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed)}"
            )
        
        # Read file
        content = await file.read()
        
        # Size check (100MB)
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 100MB limit")
        
        logger.info(f"Processing: {file.filename} ({len(content)} bytes)")
        
        # Extract text
        text = extract_text_from_file(content, file.filename)
        
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract text from {file.filename}. The file may be empty, corrupted, or contain only images. Please try a different file."
            )
        
        logger.info(f"Extracted {len(text)} characters")
        logger.info(f"Sample: {text[:200]}...")
        
        # Create chunks
        chunks = create_chunks(text, chunk_size=600, overlap=150)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create chunks")
        
        # Store
        doc_id = file.filename
        document_store[doc_id] = chunks
        
        if doc_id not in uploaded_files:
            uploaded_files.append(doc_id)
        
        logger.info(f"✓ Stored {len(chunks)} chunks for {doc_id}")
        
        return UploadResponse(
            message=f"Successfully processed '{file.filename}'",
            filename=file.filename,
            chunks_created=len(chunks),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Status endpoint"""
    total_chunks = sum(len(chunks) for chunks in document_store.values())
    
    return JSONResponse({
        "status": "online",
        "documents_loaded": len(document_store),
        "total_chunks": total_chunks,
        "document_names": uploaded_files,
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "openrouter_configured": bool(os.getenv('OPENROUTER_API_KEY'))
    })

@router.get("/documents")
async def list_documents():
    """List documents"""
    docs = []
    for doc_id in uploaded_files:
        if doc_id in document_store:
            chunks = document_store[doc_id]
            docs.append({
                "filename": doc_id,
                "chunks": len(chunks)
            })
    
    return JSONResponse({"documents": docs, "count": len(docs)})

@router.delete("/clear")
async def clear_all():
    """Clear all documents"""
    document_store.clear()
    uploaded_files.clear()
    logger.info("Cleared all documents")
    return JSONResponse({"message": "All documents cleared", "status": "success"})
