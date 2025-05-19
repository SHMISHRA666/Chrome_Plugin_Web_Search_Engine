# Web Search Engine Chrome Extension

A Chrome extension that builds a semantic search index of your browsing history using Nomic embeddings and FAISS.

## Features

- Automatically indexes web pages you visit (skips private/confidential sites)
- Uses Nomic embeddings for semantic search
- FAISS for efficient similarity search
- Highlights matching text when viewing search results
- Beautiful and intuitive user interface

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key
```

3. Start the backend server:
```bash
python server.py
```

4. Load the Chrome extension:
- Open Chrome and go to `chrome://extensions/`
- Enable "Developer mode"
- Click "Load unpacked"
- Select the extension directory

## Usage

1. Browse the web normally - the extension will automatically index pages you visit
2. Click the extension icon to open the search interface
3. Enter your search query
4. Click on a result to open the page with the matching text highlighted

## Architecture

The project follows a layered architecture:

- **Models Layer**: Defines data structures using Pydantic
- **Perception Layer**: Handles webpage content extraction
- **Memory Layer**: Manages FAISS index and embeddings
- **Decision Layer**: Processes search queries and ranks results
- **Action Layer**: Handles Chrome extension actions
- **Agent Layer**: Orchestrates the entire system

## Development

- Backend: Python with Flask
- Frontend: HTML/CSS/JavaScript
- Search: FAISS + Nomic embeddings
- LLM: Gemini Flash 2.0

## License

MIT License
