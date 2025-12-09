"""
FastAPI application entry point.
Initializes the app, middleware, and routes.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="A chat application with Retrieval-Augmented Generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get directories
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Import and include routers
from app.api.chat import router as chat_router
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Root endpoints
@app.get("/")
async def read_root():
    """Serve the main page"""
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Welcome to RAG Chat API", "docs": "/docs"}

@app.get("/chat")
async def read_chat():
    """Serve the chat interface"""
    chat_path = TEMPLATES_DIR / "chat.html"
    if chat_path.exists():
        return FileResponse(str(chat_path))
    return {"message": "Chat interface not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}