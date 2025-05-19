import asyncio
import time
import os
import datetime
from perception import extract_perception, extract_content
from memory import MemoryManager
from decision import generate_plan, process_search_query
from action import execute_tool, save_to_index, highlight_text, format_search_results
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

    async def process_page(self, url: str, session) -> bool:
        """Process a webpage and add to index."""
        try:
            # # Check if URL is already indexed
            # if hasattr(self.memory, 'data'):
            #     for item in self.memory.data:
            #         if item.url == url:
            #             log("agent", f"URL {url} already indexed, skipping processing.")
            #             return True

            # Extract content
            perception = extract_content(url)
            if not perception.is_indexable:
                log("agent", f"Skipping {url}: {perception.skip_reason}")
                return False
                
            # Create webpage content
            content = MemoryItem(
                url=url,
                title=perception.title,
                content=perception.content,
                type="fact",
                session_id=self.session_id
            )
            
            save_to_index(content,self.index_path)

            # Extract content if not already done
            if not content.title or not content.content:
                perception = extract_content(content.url)
                content.title = perception.title
                content.content = perception.content

            try:
                # Now call the tool with all required fields
                print("Calling MCP tool process_webpage_tool")
                result = await session.call_tool(
                    "process_webpage_tool",
                    arguments={
                        "data": {
                            "url": content.url,
                            "title": content.title,
                            "content": content.content
                        }
                    }
                )
                # print(result)
                if not result:
                    log("agent", f"Failed to process {url} with MCP tool - no result returned")
                    return False
                
            except Exception as tool_error:
                log("agent", f"Tool execution failed for {url}: {str(tool_error)}")
                import traceback
                log("agent", f"Tool error traceback: {traceback.format_exc()}")
                return False
            
            # Add to memory
            self.memory.add(content)
            log("agent", f"Processed {url}")
            return True

        except Exception as e:
            log("agent", f"Failed to process {url}: {str(e)}")
            import traceback
            log("agent", f"Error traceback: {traceback.format_exc()}")
            return False

    async def process_query(self, query: str, session, tools) -> dict:
        """Process a user query through the agent loop."""
        try:
            step = 0
            original_query = query
            context_found = False
            final_answer = None
            session_id = f"session-{int(time.time())}"

            while step < self.max_steps and not context_found:
                log("loop", f"Step {step + 1} started")

                # Extract perception
                perception = extract_perception(query)
                log("perception", f"Intent: {perception.intent}, Tool hint: {perception.tool_hint}")

                # Create search query object
                search_query = SearchQuery(query=query, top_k=3, session_filter=session_id)
                retrieved = self.memory.search(search_query)
                log("memory", f"Retrieved {len(retrieved.results)} relevant memories")

                # Get tool descriptions
                tool_descriptions = "\n".join(
                    f"- {tool.name}: {getattr(tool, 'description', 'No description')}" 
                    for tool in tools.tools
                )

                # Generate plan
                plan = generate_plan(perception, retrieved.results, tool_descriptions=tool_descriptions)
                log("plan", f"Plan generated: {plan}")

                if plan.startswith("NO_TOOL_NEEDED:"):
                    context_found = True
                    final_answer = plan.replace("NO_TOOL_NEEDED:", "").strip()
                    break

                if plan.startswith("RELEVANT_CONTEXT_FOUND:"):
                    context_found = True
                    search_results = process_search_query(original_query, self.memory, top_k=5, plan_result=plan)
                    final_answer = search_results.final_answer
                    break

                try:
                    # Execute tool
                    result = await execute_tool(session, tools.tools, plan)
                    log("tool", f"{result.tool_name} returned: {result.result}")

                    # Add to memory with proper fields
                    memory_item = MemoryItem(
                        text=str(result.result),
                        type="tool_output",
                        tool_name=result.tool_name,
                        user_query=original_query,
                        tags=[result.tool_name],
                        session_id=self.session_id,
                        url="",  # Required field
                        title="",  # Required field
                        content=str(result.result)  # Required field
                    )
                    self.memory.add(memory_item)

                    # Update query for next iteration
                    query = f"Original task: {original_query}\nPrevious output: {result.result}\nWhat should I do next?"

                except Exception as e:
                    log("error", f"Tool execution failed: {e}")
                    break

                step += 1

            # If no final answer yet, process search results
            if not final_answer:
                search_results = process_search_query(original_query, self.memory, top_k=5, plan_result=plan)
                final_answer = search_results.final_answer

            return {
                "success": True,
                "answer": final_answer,
                "search_results": search_results.model_dump() if 'search_results' in locals() else None
            }

        except Exception as e:
            log("error", f"Query processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

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
                
                if not url:
                    return jsonify({'error': 'No URL provided'}), 400
                
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
                                        success = await self.process_page(url, session)                               
                                    except Exception as e:
                                        print(f"[agent] Error initializing MCP: {e}")
                                        raise
                            except Exception as e:
                                print(f"[agent] Error initializing MCP: {e}")
                                raise
                    except Exception as e:
                        print(f"[agent] Error establishing connection: {e}")
                        raise
                    if success:
                        return jsonify({'status': 'success'})
                    else:
                        return jsonify({'error': 'Failed to process URL'}), 500
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

        @app.route('/query', methods=['POST'])
        async def handle_query():
            """Handle user queries."""
            try:
                data = request.get_json()
                query = data.get('query')
                
                if not query:
                    return jsonify({'error': 'No query provided'}), 400
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
                                    result = await self.process_query(query, session, tools)
                                except Exception as e:
                                    print(f"[agent] Error initializing MCP: {e}")
                                    raise
                        except Exception as e:
                            print(f"[agent] Error initializing MCP: {e}")
                            raise
                except Exception as e:
                    print(f"[agent] Error establishing connection: {e}")
                    raise
                return jsonify(result)

            except Exception as e:
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