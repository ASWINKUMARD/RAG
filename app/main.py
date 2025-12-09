"""
FastAPI application entry point.
Initializes the app, middleware, and routes.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="A chat application with Retrieval-Augmented Generation",
    version="1.0.0"
)

# CORS middleware - CRITICAL FOR FRONTEND-BACKEND CONNECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # For development
        "https://rag-1-csd2.onrender.com",  # Your production domain
        "http://localhost:8000",
        "http://localhost:3000",
    ],
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

# Mount static files (if you have CSS/JS files)
if STATIC_DIR.exists() and any(STATIC_DIR.iterdir()):
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
    return JSONResponse({
        "message": "Welcome to RAG Chat API",
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "chat": "/api/chat",
            "upload": "/api/upload",
            "clear": "/api/clear"
        }
    })

@app.get("/chat")
async def read_chat():
    """Serve the chat interface"""
    chat_path = TEMPLATES_DIR / "chat.html"
    if chat_path.exists():
        return FileResponse(str(chat_path))
    return JSONResponse({"error": "Chat interface not found"}, status_code=404)

@app.get("/health")
async def health_check():
    """Health check endpoint - CRITICAL for frontend status checks"""
    return JSONResponse({
        "status": "healthy",
        "version": "1.0.0",
        "service": "RAG Chat API"
    })

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 RAG Chat API is starting...")
    print(f"📁 Templates directory: {TEMPLATES_DIR}")
    print(f"📁 Static directory: {STATIC_DIR}")
    print("✅ Server ready!")

# For local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
