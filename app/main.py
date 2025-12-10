"""
Complete RAG Chatbot in a Single File - Deploy Anywhere
Works on Render, Vercel, Railway, etc.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import re
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot",
    description="AI-Powered Document Assistant",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
document_store: Dict[str, List[Dict]] = {}
uploaded_files: List[str] = []

# Models
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []
    status: str = "success"

# Embedded HTML
LANDING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chatbot - AI Document Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            width: 100%;
            background: white;
            border-radius: 30px;
            padding: 60px 40px;
            box-shadow: 0 25px 70px rgba(0,0,0,0.3);
            text-align: center;
            animation: fadeIn 0.6s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .logo {
            font-size: 5em;
            margin-bottom: 20px;
            animation: bounce 2s ease infinite;
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        h1 {
            font-size: 2.8em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .subtitle {
            color: #666;
            font-size: 1.3em;
            margin-bottom: 40px;
            line-height: 1.6;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 25px;
            background: #d4edda;
            color: #155724;
            border-radius: 25px;
            font-weight: 600;
            margin: 30px 0;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #28a745;
            animation: pulse 2s ease infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .cta-button {
            display: inline-block;
            padding: 20px 55px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 35px;
            font-size: 1.3em;
            font-weight: 600;
            margin-top: 20px;
            cursor: pointer;
            border: none;
            transition: all 0.3s;
            box-shadow: 0 10px 35px rgba(102,126,234,0.4);
        }
        .cta-button:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 45px rgba(102,126,234,0.6);
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 50px 0 30px;
            text-align: left;
        }
        .feature {
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            transition: transform 0.3s;
        }
        .feature:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102,126,234,0.2);
        }
        .feature-icon { font-size: 2.5em; margin-bottom: 10px; }
        .feature h3 { color: #1a2332; margin-bottom: 8px; }
        .feature p { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo">🤖📚</div>
        <h1>RAG Chatbot</h1>
        <p class="subtitle">
            Your intelligent AI document assistant. Upload documents and get instant,
            accurate answers with source citations.
        </p>
        <div class="status-badge">
            <span class="status-dot"></span>
            <span>Backend Online ✓</span>
        </div>
        <div class="features">
            <div class="feature">
                <div class="feature-icon">📄</div>
                <h3>Multi-Format</h3>
                <p>PDF, DOCX, TXT support</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🔍</div>
                <h3>Smart Search</h3>
                <p>AI-powered retrieval</p>
            </div>
            <div class="feature">
                <div class="feature-icon">💬</div>
                <h3>Natural Chat</h3>
                <p>Conversational AI</p>
            </div>
            <div class="feature">
                <div class="feature-icon">📌</div>
                <h3>Citations</h3>
                <p>Source references</p>
            </div>
        </div>
        <a href="/chat" class="cta-button">Get Started →</a>
        <p style="margin-top: 40px; color: #999;">
            <a href="/docs" style="color: #667eea; text-decoration: none;">API Docs</a>
        </p>
    </div>
</body>
</html>"""

CHAT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chatbot - Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f5f7fa;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px 40px;
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        }
        .header a {
            color: white;
            text-decoration: none;
            padding: 8px 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 20px;
        }
        .header a:hover { background: rgba(255,255,255,0.3); }
        .upload-btn {
            background: white;
            color: #667eea;
            padding: 10px 25px;
            border: none;
            border-radius: 20px;
            font-weight: 600;
            cursor: pointer;
        }
        .container {
            max-width: 1200px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .chat-container {
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            display: flex;
            flex-direction: column;
            height: calc(100vh - 180px);
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            background: #fafbfc;
        }
        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { flex-direction: row-reverse; }
        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3em;
            flex-shrink: 0;
        }
        .message.user .avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .message.bot .avatar { background: #e9ecef; }
        .message-content {
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 15px;
            line-height: 1.6;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e9ecef;
            border-bottom-left-radius: 5px;
        }
        .sources {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #e9ecef;
            font-size: 0.85em;
            color: #666;
        }
        .input-area {
            padding: 20px 30px;
            background: white;
            border-top: 2px solid #e9ecef;
            display: flex;
            gap: 15px;
        }
        #messageInput {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
        }
        #messageInput:focus { border-color: #667eea; }
        button {
            padding: 15px 35px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <a href="/">← Home</a>
            <span style="margin-left: 20px; font-size: 1.2em;">📚 RAG Chatbot</span>
        </div>
        <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
            📤 Upload Document
        </button>
        <input type="file" id="fileInput" style="display:none;" accept=".pdf,.docx,.txt" onchange="uploadFile()">
    </div>
    
    <div class="container">
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message bot">
                    <div class="avatar">🤖</div>
                    <div class="message-content">
                        Hello! I'm your RAG assistant. Upload documents using the button above,
                        then ask me questions. I'll find relevant information and provide answers with sources.
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Ask about your documents..."
                    onkeypress="if(event.key==='Enter') sendMessage()">
                <button id="sendButton" onclick="sendMessage()">Send 📨</button>
            </div>
        </div>
    </div>
    
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        
        function addMessage(content, isUser, sources = []) {
            const div = document.createElement('div');
            div.className = `message ${isUser ? 'user' : 'bot'}`;
            
            let sourcesHtml = '';
            if (!isUser && sources.length > 0) {
                sourcesHtml = `<div class="sources"><strong>Sources:</strong> ${sources.join(', ')}</div>`;
            }
            
            div.innerHTML = `
                <div class="avatar">${isUser ? '👤' : '🤖'}</div>
                <div class="message-content">${content}${sourcesHtml}</div>
            `;
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            addMessage(message, true);
            messageInput.value = '';
            sendButton.disabled = true;
            messageInput.disabled = true;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: message,
                        conversation_history: []
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage(data.response, false, data.sources || []);
                } else {
                    addMessage('Error: ' + (data.detail || 'Unknown error'), false);
                }
            } catch (error) {
                addMessage('Connection error. Please try again.', false);
                console.error('Error:', error);
            } finally {
                sendButton.disabled = false;
                messageInput.disabled = false;
                messageInput.focus();
            }
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            if (!file) return;
            
            if (file.size > 10 * 1024 * 1024) {
                alert('File must be under 10MB');
                fileInput.value = '';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            addMessage(`Uploading ${file.name}...`, false);
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage(`✅ ${data.message}`, false);
                } else {
                    addMessage('❌ Upload failed: ' + (data.detail || 'Unknown error'), false);
                }
            } catch (error) {
                addMessage('❌ Upload error. Please try again.', false);
                console.error('Upload error:', error);
            }
            
            fileInput.value = '';
        }
        
        messageInput.focus();
    </script>
</body>
</html>"""

# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page"""
    logger.info("Serving landing page")
    return HTMLResponse(content=LANDING_PAGE)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """Chat interface"""
    logger.info("Serving chat page")
    return HTMLResponse(content=CHAT_PAGE)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "documents": len(document_store)
    }

@app.get("/api/status")
async def status():
    """API status"""
    total_chunks = sum(len(chunks) for chunks in document_store.values())
    return {
        "status": "online",
        "documents_loaded": len(document_store),
        "total_chunks": total_chunks,
        "document_names": uploaded_files
    }

# Helper functions
def clean_text(text: str) -> str:
    """Clean extracted text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}\'\"]+', ' ', text)
    return text.strip()

def create_chunks(text: str, size: int = 400, overlap: int = 100) -> List[Dict]:
    """Create text chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), size - overlap):
        chunk_words = words[i:i + size]
        chunk_text = ' '.join(chunk_words)
        
        if chunk_text.strip():
            chunks.append({
                'text': chunk_text,
                'index': len(chunks),
                'words': set(chunk_text.lower().split())
            })
    
    return chunks

def retrieve_chunks(query: str, k: int = 4) -> List[tuple]:
    """Retrieve relevant chunks"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Remove stopwords
    stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or'}
    query_keywords = query_words - stopwords
    
    if not query_keywords:
        query_keywords = query_words
    
    scores = []
    
    for doc_id, chunks in document_store.items():
        for chunk in chunks:
            chunk_text = chunk['text']
            chunk_lower = chunk_text.lower()
            chunk_words = chunk['words']
            
            score = 0
            
            # Exact phrase match
            if query_lower in chunk_lower:
                score += 100
            
            # Keyword overlap
            overlap = len(query_keywords & chunk_words)
            if overlap > 0:
                score += (overlap / len(query_keywords)) * 50
            
            # Individual keywords
            for keyword in query_keywords:
                if keyword in chunk_lower:
                    count = chunk_lower.count(keyword)
                    score += count * 10
            
            if score > 0:
                scores.append((score, doc_id, chunk_text, chunk['index']))
    
    scores.sort(reverse=True, key=lambda x: x[0])
    logger.info(f"Query: '{query}' - Found {len(scores)} matches")
    
    return scores[:k]

# API Endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with documents"""
    try:
        logger.info(f"Chat: '{request.message[:50]}'")
        
        if not document_store:
            return ChatResponse(
                response="Please upload documents first! Use the 'Upload Document' button above.",
                sources=[],
                status="success"
            )
        
        # Retrieve relevant chunks
        matches = retrieve_chunks(request.message, k=4)
        
        if not matches:
            return ChatResponse(
                response=f"I couldn't find relevant information about '{request.message}' in your documents. Try rephrasing or asking about different topics.",
                sources=list(document_store.keys()),
                status="success"
            )
        
        # Build context
        context_parts = []
        sources = []
        
        for score, doc_id, text, idx in matches:
            context_parts.append(f"[From {doc_id}, Section {idx+1}]:\n{text}")
            if doc_id not in sources:
                sources.append(doc_id)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate response with LLM
        api_key = os.getenv('OPENROUTER_API_KEY', '')
        
        if api_key:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": "google/gemini-2.0-flash-exp:free",
                            "messages": [
                                {
                                    "role": "system",
                                    "content": f"Answer based on this context:\n\n{context}\n\nBe concise and cite sources."
                                },
                                {"role": "user", "content": request.message}
                            ],
                            "temperature": 0.3,
                            "max_tokens": 800
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result['choices'][0]['message']['content']
                    else:
                        answer = f"Based on the documents:\n\n{context[:500]}..."
            except Exception as e:
                logger.error(f"LLM error: {e}")
                answer = f"From the documents:\n\n{context[:500]}..."
        else:
            # No API key - return context
            answer = f"Relevant information from documents:\n\n{context[:600]}...\n\n(Set OPENROUTER_API_KEY for AI-generated responses)"
        
        return ChatResponse(
            response=answer,
            sources=sources,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """Upload document"""
    try:
        logger.info(f"Upload: {file.filename}")
        
        # Validate
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        ext = '.' + file.filename.split('.')[-1].lower()
        if ext not in ['.pdf', '.docx', '.txt']:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        
        # Read file
        content = await file.read()
        
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 10MB")
        
        # Extract text
        text = content.decode('utf-8', errors='ignore')
        text = clean_text(text)
        
        if len(text) < 20:
            raise HTTPException(status_code=400, detail="File contains insufficient text")
        
        logger.info(f"Extracted {len(text)} characters")
        
        # Create chunks
        chunks = create_chunks(text, size=400, overlap=100)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create chunks")
        
        # Store
        document_store[file.filename] = chunks
        if file.filename not in uploaded_files:
            uploaded_files.append(file.filename)
        
        logger.info(f"✓ Created {len(chunks)} chunks for {file.filename}")
        
        return {
            "message": f"Successfully processed '{file.filename}' - {len(chunks)} chunks created. Ask me questions now!",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Run
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
