from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import os
import io
import httpx
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

document_store: Dict[str, List[Dict]] = {}
uploaded_files = []

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
    """Extract text from PDF with multiple fallback methods"""
    text = ""
    
    # Method 1: PyPDF2
    try:
        import PyPDF2
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"PyPDF2 page {page_num} error: {e}")
        
        text = '\n\n'.join(text_parts)
        if len(text.strip()) > 100:
            logger.info(f"✓ PyPDF2 extracted {len(text)} chars")
            return text
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}")
    
    # Method 2: pdfplumber (more robust for complex PDFs)
    try:
        import pdfplumber
        pdf_file = io.BytesIO(content)
        
        text_parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"pdfplumber page {page_num} error: {e}")
        
        text = '\n\n'.join(text_parts)
        if len(text.strip()) > 100:
            logger.info(f"✓ pdfplumber extracted {len(text)} chars")
            return text
    except ImportError:
        logger.info("pdfplumber not available")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    
    # Method 3: pypdf (alternative library)
    try:
        import pypdf
        pdf_file = io.BytesIO(content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        
        text_parts = []
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                logger.warning(f"pypdf page error: {e}")
        
        text = '\n\n'.join(text_parts)
        if len(text.strip()) > 100:
            logger.info(f"✓ pypdf extracted {len(text)} chars")
            return text
    except ImportError:
        logger.info("pypdf not available")
    except Exception as e:
        logger.warning(f"pypdf failed: {e}")
    
    return text

def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX"""
    try:
        import docx
        
        docx_file = io.BytesIO(content)
        doc = docx.Document(docx_file)
        
        text_parts = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text_parts.append(cell.text)
        
        full_text = '\n\n'.join(text_parts)
        
        if len(full_text.strip()) > 20:
            logger.info(f"✓ Extracted {len(full_text)} chars from DOCX")
            return full_text
        return ""
            
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="python-docx not installed. Add 'python-docx==1.1.0' to requirements.txt"
        )
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_file(content: bytes, filename: str) -> str:
    """Main text extraction with improved error handling"""
    try:
        file_lower = filename.lower()
        
        if file_lower.endswith('.txt'):
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text = content.decode(encoding)
                    if len(text.strip()) > 20:
                        logger.info(f"✓ TXT decoded with {encoding}: {len(text)} chars")
                        return text
                except:
                    continue
            return ""
        
        elif file_lower.endswith('.pdf'):
            text = extract_text_from_pdf(content)
            # Clean up PDF text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text.strip()
        
        elif file_lower.endswith('.docx'):
            return extract_text_from_docx(content)
        
        else:
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Text extraction error for {filename}: {e}")
        return ""

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """Create overlapping text chunks"""
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text or len(text) < 50:
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        
        if current_length + word_count > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'index': chunk_idx,
                'words': set(chunk_text.lower().split())
            })
            chunk_idx += 1
            
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + words
                current_length = len(current_chunk)
            else:
                current_chunk = words
                current_length = word_count
        else:
            current_chunk.extend(words)
            current_length += word_count
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'index': chunk_idx,
            'words': set(chunk_text.lower().split())
        })
    
    logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
    return chunks

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[tuple]:
    """Retrieve relevant chunks using keyword matching"""
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 
                  'but', 'in', 'with', 'to', 'for', 'of', 'it', 'as', 'by'}
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
            
            if query_lower in chunk_lower:
                score += 300
            
            if query_keywords.issubset(chunk_words):
                score += 150
            
            overlap = len(query_keywords & chunk_words)
            if overlap > 0:
                score += (overlap / len(query_keywords)) * 100
            
            for keyword in query_keywords:
                count = chunk_lower.count(keyword)
                score += count * 30
            
            if score > 0:
                results.append((score, doc_id, chunk_text, chunk['index']))
    
    results.sort(reverse=True, key=lambda x: x[0])
    
    logger.info(f"Found {len(results)} chunks (top score: {results[0][0] if results else 0})")
    return results[:top_k]

async def generate_llm_response(query: str, context: str) -> str:
    """Generate response using OpenRouter API"""
    
    system_prompt = f"""You are a helpful RAG assistant. Answer based on the provided context.

CONTEXT:
{context}

RULES:
- Answer using ONLY information from the context
- Be specific and detailed
- Use bullet points for clarity
- If information is missing, say so
- Reference document sections when relevant"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        if not api_key:
            logger.warning("No OPENROUTER_API_KEY - using context directly")
            return f"Based on the documents:\n\n{context[:800]}\n\n(Set OPENROUTER_API_KEY for AI responses)"
        
        async with httpx.AsyncClient(timeout=50.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://rag-chatbot.com",
                    "X-Title": "RAG Chatbot"
                },
                json={
                    "model": "kwaipilot/kat-coder-pro:free",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1200
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenRouter error: {response.status_code}")
                return f"Based on the documents:\n\n{context[:800]}"
                
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"Based on the documents:\n\n{context[:800]}"

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    try:
        logger.info(f"Query: {request.message}")
        
        if not document_store:
            return ChatResponse(
                response="Please upload documents first! Use the 'Upload Document' button above.",
                sources=[],
                status="success"
            )
        
        retrieved = retrieve_relevant_chunks(request.message, top_k=5)
        
        if not retrieved:
            return ChatResponse(
                response=f"No relevant information found for '{request.message}'. Try different keywords or upload more documents.",
                sources=list(document_store.keys()),
                status="success"
            )
        
        context_parts = []
        sources = set()
        
        for score, doc_id, chunk_text, chunk_idx in retrieved:
            context_parts.append(f"[{doc_id} - Section {chunk_idx + 1}]\n{chunk_text}")
            sources.add(doc_id)
        
        context = "\n\n---\n\n".join(context_parts)
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
    """Upload document with robust error handling"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        allowed = ['.pdf', '.docx', '.txt']
        ext = '.' + file.filename.rsplit('.', 1)[-1].lower()
        
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"File type '{ext}' not supported. Use: PDF, DOCX, or TXT"
            )
        
        content = await file.read()
        
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 100MB")
        
        if len(content) < 100:
            raise HTTPException(status_code=400, detail="File too small or empty")
        
        logger.info(f"Processing: {file.filename} ({len(content)} bytes)")
        
        text = extract_text_from_file(content, file.filename)
        
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract text from '{file.filename}'. "
                       f"Possible reasons: (1) PDF contains only images/scans, "
                       f"(2) File is corrupted, (3) File is password-protected. "
                       f"Try: Converting scanned PDFs with OCR, using a different file format, "
                       f"or ensuring the file opens correctly on your computer."
            )
        
        logger.info(f"Extracted {len(text)} characters")
        
        chunks = create_chunks(text, chunk_size=500, overlap=100)
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="Text extracted but chunking failed. File may be too short."
            )
        
        doc_id = file.filename
        document_store[doc_id] = chunks
        
        if doc_id not in uploaded_files:
            uploaded_files.append(doc_id)
        
        logger.info(f"✓ Successfully stored {len(chunks)} chunks for '{doc_id}'")
        
        return UploadResponse(
            message=f"Successfully processed '{file.filename}' with {len(chunks)} chunks",
            filename=file.filename,
            chunks_created=len(chunks),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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
    """List uploaded documents"""
    docs = []
    for doc_id in uploaded_files:
        if doc_id in document_store:
            docs.append({
                "filename": doc_id,
                "chunks": len(document_store[doc_id])
            })
    
    return JSONResponse({"documents": docs, "count": len(docs)})

@router.delete("/clear")
async def clear_all():
    """Clear all documents"""
    document_store.clear()
    uploaded_files.clear()
    logger.info("All documents cleared")
    return JSONResponse({"message": "All documents cleared", "status": "success"})
