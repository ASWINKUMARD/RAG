# RAG Chat Application

A FastAPI-based chat application with Retrieval-Augmented Generation (RAG) capabilities.

## Features

- 💬 Interactive chat interface
- 📄 Document upload and processing
- 🧠 AI-powered responses using RAG
- 🔍 Semantic search with embeddings
- 🚀 Fast and scalable with FastAPI

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **AI/ML**: Sentence Transformers, HuggingFace
- **Vector DB**: ChromaDB / FAISS
- **Frontend**: HTML, CSS, JavaScript

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/rag-chat-app.git
cd rag-chat-app
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
# Add your API keys
```

5. Run the application:
```bash
python run.py
```

6. Open browser:
```
http://localhost:8000
```

## Project Structure
```
project_root/
├── app/                # Main application
│   ├── api/           # API endpoints
│   ├── models/        # Data models
│   ├── services/      # Business logic
│   └── utils/         # Utilities
├── static/            # CSS, JS, images
├── templates/         # HTML templates
├── data/              # Data storage
└── tests/             # Tests
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Usage

### Upload Documents
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf"
```

### Chat
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Your question here"}'
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

MIT License

## Contact

Your Name - Aswin Kumar D

