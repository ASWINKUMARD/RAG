"""
FastAPI main entry point for RAG Chatbot
Serves frontend + connects chat routes using OpenRouter model RAG backend.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Directory Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create folders if not present
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# -----------------------------
# Lifespan Context Manager (Replaces deprecated on_event)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("\n🚀 RAG Chatbot Server Starting...")
    logger.info(f"📂 Templates → {TEMPLATES_DIR}")
    logger.info(f"📂 Static    → {STATIC_DIR}")
    logger.info("🧠 Model     → kwaipilot/kat-coder-pro:free via OpenRouter")
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY', '')
    if api_key:
        logger.info("✅ OPENROUTER_API_KEY configured")
    else:
        logger.warning("⚠️  OPENROUTER_API_KEY not set - responses will be basic")
    
    logger.info("----------------------------------------\n")
    
    yield  # Server runs here
    
    # Shutdown
    logger.info("\n🛑 RAG Chatbot Server Shutting Down...")

# -----------------------------
# Initialize FastAPI App
# -----------------------------
app = FastAPI(
    title="RAG Chat API",
    description="AI Chatbot with RAG using OpenRouter + kwaipilot/kat-coder-pro",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------
# CORS (Required for Frontend to call Backend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all for development
        "https://rag-1-csd2.onrender.com",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Static Files Mount
# -----------------------------
if STATIC_DIR.exists() and any(STATIC_DIR.iterdir()):
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info(f"✅ Static files mounted from {STATIC_DIR}")

# -----------------------------
# ROUTES IMPORT
# Import chat router from app/api/chat.py
# -----------------------------
try:
    from app.api.chat import router as chat_router
    app.include_router(chat_router, prefix="/api", tags=["RAG-Chat"])
    logger.info("✅ Chat router loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import chat router: {e}")
    logger.error("Make sure the file structure is: app/api/chat.py")

# -----------------------------
# FRONTEND ROUTES
# -----------------------------
@app.get("/")
async def home():
    """Serve landing UI page (index.html)"""
    index_page = TEMPLATES_DIR / "index.html"
    if index_page.exists():
        return FileResponse(str(index_page))
    
    return JSONResponse(
        content={
            "message": "RAG Chatbot API is running",
            "docs": "/docs",
            "health": "/health",
            "note": "Frontend UI missing. Add templates/index.html"
        },
        status_code=200
    )

@app.get("/chat")
async def chat_page():
    """Serve chat UI page"""
    chat_page = TEMPLATES_DIR / "chat.html"
    if chat_page.exists():
        return FileResponse(str(chat_page))
    
    return JSONResponse(
        content={"error": "chat.html not found in templates directory"},
        status_code=404
    )

@app.get("/health")
async def health():
    """Health check endpoint - frontend pings this regularly"""
    return JSONResponse(
        content={
            "status": "online",
            "version": "1.0.0",
            "service": "RAG Chatbot",
            "openrouter_configured": bool(os.getenv('OPENROUTER_API_KEY'))
        }
    )

# -----------------------------
# ERROR HANDLING
# -----------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "method": request.method
        },
        status_code=404
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        content={
            "error": "Internal server error",
            "details": str(exc),
            "path": str(request.url.path)
        },
        status_code=500
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        content={
            "error": "An unexpected error occurred",
            "type": type(exc).__name__,
            "details": str(exc)
        },
        status_code=500
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
