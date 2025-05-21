import asyncio
import time
import os
import datetime
from perception import extract_perception, extract_content
from memory import MemoryManager
from decision import generate_plan, process_search_query
from action import execute_tool, save_to_index, \
highlight_text, format_search_results, process_page, process_query
from models import MemoryItem, SearchQuery, SearchResponse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import json

def log(stage: str, msg: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{stage}] {msg}")

class WebSearchAgent:
    def __init__(self, index_path="faiss_index"):
        self.memory = MemoryManager(index_path=index_path)
        self.index_path = index_path
        self.session_id = f"session-{int(time.time())}"
        self.max_steps = 3

    def setup_flask_app(self):
        """Setup Flask app for the Google extension."""
        app = Flask(__name__)
        CORS(app)

        @app.route('/', methods=['GET'])
        def root():
            """Root endpoint that returns basic info about the server."""
            return jsonify({
                'status': 'running',
                'name': 'Web Search Assistant API',
                'version': '1.0'
            })

        @app.route('/connect', methods=['GET'])
        def connect():
            """Test connection to backend."""
            return jsonify({'status': 'connected'})

        @app.route('/process', methods=['POST'])
        async def process_url():
            """Process a URL and add to index."""
            try:
                data = request.get_json()
                url = data.get('url')
                query = data.get('query')
                
                if not url and not query:
                    return jsonify({'error': 'No URL or query provided'}), 400
                
                if url:
                    log("process", f"Attempting to process URL: {url}")
                
                try:
                    server_params = StdioServerParameters(
                        command="python",
                        args=["mcp_server.py"]
                    )
                    try:
                        async with stdio_client(server_params) as (read, write):
                            print("Connection established, creating session...")
                            try:
                                async with ClientSession(read, write) as session:
                                    print("[agent] Session created, initializing...")
                                    try:
                                        await session.initialize()
                                        print("[agent] MCP session initialized")
                                        tools = await session.list_tools()
                                        
                                        # Handle both URL processing and query in the same session
                                        if url:
                                            success = await process_page(url, session)
                                            if not success:
                                                return jsonify({'error': 'Failed to process URL'}), 500
                                        
                                        if query:
                                            result = await process_query(query, session, tools)
                                            return jsonify(result)
                                        
                                        # If only URL was processed and no query
                                        if url:
                                            return jsonify({'status': 'success'})
                                            
                                    except Exception as e:
                                        print(f"[agent] Error initializing MCP: {e}")
                                        raise
                            except Exception as e:
                                print(f"[agent] Error initializing MCP: {e}")
                                raise
                    except Exception as e:
                        print(f"[agent] Error establishing connection: {e}")
                        raise
                except Exception as e:
                    log("error", f"Error in process_page: {str(e)}")
                    import traceback
                    log("error", f"Traceback: {traceback.format_exc()}")
                    return jsonify({'error': f'Processing error: {str(e)}'}), 500
                    
            except Exception as e:
                log("error", f"Error in process_url endpoint: {str(e)}")
                import traceback
                log("error", f"Traceback: {traceback.format_exc()}")
                return jsonify({'error': str(e)}), 500

        @app.route('/stats', methods=['GET'])
        def get_stats():
            """Get index statistics."""
            try:
                stats = self.memory.get_stats()
                return jsonify(stats.model_dump())
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        return app

    def run(self):
        """Run the Flask server."""
        # Create index directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        # Setup and run Flask app
        app = self.setup_flask_app()
        
        app.run(host='localhost', port=5000)

async def main():
    # Initialize and run agent
    agent = WebSearchAgent()
    agent.run()

if __name__ == "__main__":
    asyncio.run(main()) 