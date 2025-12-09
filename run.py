"""
Main entry point to start the FastAPI server.
Run this file to start the application.
"""
import multiprocessing
import uvicorn
import sys
from pathlib import Path

def main():
    # Ensure the current directory is in Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Run uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(current_dir / "app")]
    )

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()