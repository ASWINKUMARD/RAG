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
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CRITICAL: Proper CORS configuration for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins (for development and production)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]
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
    try:
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    except Exception as e:
        print(f"Warning: Could not mount static files: {e}")

# Import and include routers
from app.api.chat import router as chat_router
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Root endpoints
@app.get("/")
async def read_root():
    """Serve the main landing page"""
    index_path = TEMPLATES_DIR / "index.html"
    
    if index_path.exists():
        return FileResponse(
            str(index_path),
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*"
            }
        )
    
    # Fallback API response
    return JSONResponse({
        "message": "Welcome to RAG Chat API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "chat_interface": "/chat",
            "api_chat": "/api/chat",
            "api_upload": "/api/upload",
            "api_status": "/api/status",
            "api_documents": "/api/documents"
        }
    })

@app.get("/chat")
async def read_chat():
    """Serve the chat interface"""
    chat_path = TEMPLATES_DIR / "chat.html"
    
    if chat_path.exists():
        return FileResponse(
            str(chat_path),
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*"
            }
        )
    
    return JSONResponse(
        {"error": "Chat interface not found", "path": str(chat_path)},
        status_code=404
    )

@app.get("/health")
async def health_check():
    """Health check endpoint - CRITICAL for frontend status checks"""
    return JSONResponse({
        "status": "healthy",
        "version": "1.0.0",
        "service": "RAG Chat API",
        "endpoints_available": [
            "/health",
            "/",
            "/chat",
            "/api/chat",
            "/api/upload",
            "/api/status",
            "/api/documents"
        ]
    })

# Serve index.html for any unmatched routes (SPA fallback)
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """Catch-all route for SPA"""
    # If it's an API route that doesn't exist, return 404
    if full_path.startswith("api/"):
        return JSONResponse(
            {"error": f"API endpoint not found: /{full_path}"},
            status_code=404
        )
    
    # Otherwise, try to serve the file or index.html
    file_path = TEMPLATES_DIR / full_path
    
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    
    # Default to index.html for SPA routing
    index_path = TEMPLATES_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    
    return JSONResponse(
        {"error": "Page not found", "path": full_path},
        status_code=404
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found",
            "path": str(request.url.path),
            "available_endpoints": ["/", "/chat", "/health", "/api/status"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("🚀 RAG Chat API is starting...")
    print(f"📁 Base directory: {BASE_DIR}")
    print(f"📁 Templates directory: {TEMPLATES_DIR}")
    print(f"📁 Static directory: {STATIC_DIR}")
    print(f"🌐 Environment: {'Production' if not app.debug else 'Development'}")
    
    # Check if template files exist
    index_exists = (TEMPLATES_DIR / "index.html").exists()
    chat_exists = (TEMPLATES_DIR / "chat.html").exists()
    
    print(f"✓ index.html: {'Found' if index_exists else 'NOT FOUND'}")
    print(f"✓ chat.html: {'Found' if chat_exists else 'NOT FOUND'}")
    
    if not index_exists or not chat_exists:
        print("⚠️  WARNING: Template files missing!")
        print(f"   Please ensure HTML files are in: {TEMPLATES_DIR}")
    
    print("✅ Server ready!")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    print("👋 RAG Chat API is shutting down...")

# For local development
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
