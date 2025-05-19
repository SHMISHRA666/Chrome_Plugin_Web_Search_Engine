from typing import Dict, Any, Union
from pydantic import BaseModel
import json
import os
from models import MemoryItem, SearchResponse
from typing import Optional
import ast
from mcp import ClientSession

# Optional: import log from agent if shared, else define locally
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

class ToolCallResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Union[str, list, dict]
    raw_response: Any


def parse_function_call(response: str) -> tuple[str, Dict[str, Any]]:
    """Parses FUNCTION_CALL string into tool name and arguments."""
    try:
        if not response.startswith("FUNCTION_CALL:"):
            raise ValueError("Not a valid FUNCTION_CALL")

        _, function_info = response.split(":", 1)
        parts = [p.strip() for p in function_info.split("|")]
        func_name, param_parts = parts[0], parts[1:]

        result = {}
        for part in param_parts:
            if "=" not in part:
                raise ValueError(f"Invalid param: {part}")
            key, value = part.split("=", 1)

            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                parsed_value = value.strip()

            # Handle nested keys
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = parsed_value

        log("parser", f"Parsed: {func_name} → {result}")
        return func_name, result

    except Exception as e:
        log("parser", f"❌ Failed to parse FUNCTION_CALL: {e}")
        raise


async def execute_tool(session: ClientSession, tools: list[Any], response: str) -> ToolCallResult:
    """Executes a FUNCTION_CALL via MCP tool session."""
    try:
        tool_name, arguments = parse_function_call(response)

        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registered tools")

        log("tool", f"⚙️ Calling '{tool_name}' with: {arguments}")
        result = await session.call_tool(tool_name, arguments=arguments)

        if hasattr(result, 'content'):
            if isinstance(result.content, list):
                out = [getattr(item, 'text', str(item)) for item in result.content]
            else:
                out = getattr(result.content, 'text', str(result.content))
        else:
            out = str(result)

        log("tool", f"✅ {tool_name} result: {out}")
        return ToolCallResult(
            tool_name=tool_name,
            arguments=arguments,
            result=out,
            raw_response=result
        )

    except Exception as e:
        log("tool", f"⚠️ Execution failed for '{response}': {e}")
        raise

class ActionResult(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

def save_to_index(content: MemoryItem, index_path: str) -> ActionResult:
    """Save webpage content to index."""
    try:
        os.makedirs(index_path, exist_ok=True)
        data_file = os.path.join(index_path, "data.json")
        index_file = os.path.join(index_path, "index.bin")
        metadata_file = os.path.join(index_path, "metadata.json")
        cache_file = os.path.join(index_path, "webpage_cache.json")
        
        # Load existing data
        existing_data = []
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                existing_data = json.load(f)
        
        # Check if URL already exists
        url_exists = any(item.get('url') == content.url for item in existing_data)
        
        if not url_exists:
            # Add new content only if URL doesn't exist
            existing_data.append(content.model_dump())
            
            # Save updated data
            with open(data_file, 'w') as f:
                json.dump(existing_data, f)
                
            # Initialize or update FAISS index files if they don't exist
            if not all(os.path.exists(f) for f in [index_file, metadata_file, cache_file]):
                # Create empty files
                with open(metadata_file, 'w') as f:
                    json.dump([], f)
                with open(cache_file, 'w') as f:
                    json.dump({}, f)
                # Create empty FAISS index
                import faiss
                import numpy as np
                dim = 768  # Default dimension for nomic-embed-text
                index = faiss.IndexFlatL2(dim)
                faiss.write_index(index, index_file)
                
            return ActionResult(
                success=True,
                message=f"Saved content from {content.url}",
                data={"url": content.url}
            )
        else:
            return ActionResult(
                success=True,
                message=f"Content from {content.url} already exists in index",
                data={"url": content.url}
            )
        
    except Exception as e:
        log("action", f"Failed to save content: {e}")
        return ActionResult(
            success=False,
            message=str(e)
        )

def highlight_text(url: str, start: int, end: int) -> ActionResult:
    """Generate JavaScript to highlight text on webpage."""
    try:
        highlight_js = f"""
        (function() {{
            const range = document.createRange();
            const selection = window.getSelection();
            
            // Find text node containing the target position
            function findTextNode(node, targetPos) {{
                if (node.nodeType === Node.TEXT_NODE) {{
                    if (node.length >= targetPos) {{
                        return node;
                    }}
                    targetPos -= node.length;
                }}
                
                for (let child of node.childNodes) {{
                    const result = findTextNode(child, targetPos);
                    if (result) return result;
                }}
                return null;
            }}
            
            // Find start and end nodes
            const startNode = findTextNode(document.body, {start});
            const endNode = findTextNode(document.body, {end});
            
            if (startNode && endNode) {{
                range.setStart(startNode, {start});
                range.setEnd(endNode, {end});
                
                // Create highlight
                const span = document.createElement('span');
                span.style.backgroundColor = 'yellow';
                range.surroundContents(span);
                
                // Scroll to highlight
                span.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            }}
        }})();
        """
        
        return ActionResult(
            success=True,
            message="Generated highlight script",
            data={"script": highlight_js}
        )
        
    except Exception as e:
        log("action", f"Failed to generate highlight: {e}")
        return ActionResult(
            success=False,
            message=str(e)
        )

def format_search_results(results: SearchResponse) -> ActionResult:
    """Format search results for display in extension popup."""
    try:
        formatted_results = []
        for result in results.results:
            formatted_results.append({
                "url": result.url,
                "title": result.title,
                "preview": result.content[max(0, result.highlight_start-50):
                                        min(len(result.content), result.highlight_end+50)],
                "score": result.score,
                "highlight": {
                    "start": result.highlight_start,
                    "end": result.highlight_end
                }
            })
            
        return ActionResult(
            success=True,
            message=f"Found {results.total_matches} results",
            data={
                "results": formatted_results,
                "total": results.total_matches
            }
        )
        
    except Exception as e:
        log("action", f"Failed to format results: {e}")
        return ActionResult(
            success=False,
            message=str(e)
        ) 