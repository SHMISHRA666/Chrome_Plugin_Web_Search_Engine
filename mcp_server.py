from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
import sys
import os
import json
import faiss
import math
import numpy as np
from pathlib import Path
import requests
from models import \
AddInput, AddOutput, SqrtInput, SqrtOutput, StringsToIntsInput, \
StringsToIntsOutput, ExpSumInput, ExpSumOutput, \
FibonacciInput, FibonacciOutput, RemainderInput, \
RemainderOutput, SinInput, SinOutput, CosInput, CosOutput, \
TanInput, TanOutput, MineInput, MineOutput, SubtractInput, \
SubtractOutput, MultiplyInput, MultiplyOutput, DivideInput, \
DivideOutput, PowerInput, PowerOutput, CubeRootInput, CubeRootOutput, \
FactorialInput, FactorialOutput, LogInput, LogOutput
from markitdown import MarkItDown
import time
from tqdm import tqdm
import hashlib
from datetime import datetime
from pydantic import BaseModel

# Initialize MCP server
mcp = FastMCP("WebSearchEngine")

# Constants
EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40
ROOT = Path(__file__).parent.resolve()

def get_embedding(text: str) -> np.ndarray:
    response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level}: {message}\n")
    sys.stderr.flush()

@mcp.tool()
def search_pages(query: str) -> list[str]:
    """Search for relevant content from indexed webpages."""
    mcp_log("SEARCH", f"Query: {query}")
    try:
        # Validate index existence
        index_path = ROOT / "faiss_index" / "index.bin"
        meta_path = ROOT / "faiss_index" / "metadata.json"
        
        if not (index_path.exists() and meta_path.exists()):
            mcp_log("ERROR", "No index found. Please process some webpages first.")
            return ["No indexed content available for search."]
            
        # Load index and metadata
        index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text())
        
        if index.ntotal == 0 or not metadata:
            mcp_log("ERROR", "Index is empty.")
            return ["No indexed content available for search."]
            
        # Perform search
        query_vec = get_embedding(query).reshape(1, -1)
        D, I = index.search(query_vec, k=5)
        
        # Format results
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= len(metadata):
                continue
                
            data = metadata[idx]
            if not data.get('chunk'):
                continue
                
            # Add context around the match
            chunk = data['chunk']
            # preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            preview = chunk
            
            results.append(f"{preview}\n[Source: {data['url']}, Title: {data['title']}, Score: {score:.2f}]")
                
        if not results:
            return ["No matching content found for your query."]
            
        return results
        
    except Exception as e:
        mcp_log("ERROR", f"Search failed: {e}")
        return [f"Error during search: {str(e)}"]

@mcp.tool()
def add(input: AddInput) -> AddOutput:
    """Add two numbers"""
    print("CALLED: add(AddInput) -> AddOutput")
    return AddOutput(result=input.a + input.b)

@mcp.tool()
def sqrt(input: SqrtInput) -> SqrtOutput:
    """Square root of a number"""
    print("CALLED: sqrt(SqrtInput) -> SqrtOutput")
    return SqrtOutput(result=input.a ** 0.5)

# subtraction tool
@mcp.tool()
def subtract(input: SubtractInput) -> SubtractOutput:
    """Subtract two numbers"""
    print("CALLED: subtract(SubtractInput) -> SubtractOutput")
    return SubtractOutput(result=input.a - input.b)

# multiplication tool
@mcp.tool()
def multiply(input: MultiplyInput) -> MultiplyOutput:
    """Multiply two numbers"""
    print("CALLED: multiply(MultiplyInput) -> MultiplyOutput")
    return MultiplyOutput(result=input.a * input.b)

#  division tool
@mcp.tool() 
def divide(input: DivideInput) -> DivideOutput:
    """Divide two numbers"""
    print("CALLED: divide(DivideInput) -> DivideOutput")
    return DivideOutput(result=input.a / input.b)

# power tool
@mcp.tool()
def power(input: PowerInput) -> PowerOutput:
    """Power of two numbers"""
    print("CALLED: power(PowerInput) -> PowerOutput")
    return PowerOutput(result=input.a ** input.b)


# cube root tool
@mcp.tool()
def cbrt(input: CubeRootInput) -> CubeRootOutput:
    """Cube root of a number"""
    print("CALLED: cbrt(CubeRootInput) -> CubeRootOutput")
    return CubeRootOutput(result=input.a ** (1/3))

# factorial tool
@mcp.tool()
def factorial(input: FactorialInput) -> FactorialOutput:
    """factorial of a number"""
    print("CALLED: factorial(FactorialInput) -> FactorialOutput")
    return FactorialOutput(result=math.factorial(input.a))

# log tool
@mcp.tool()
def log(input: LogInput) -> LogOutput:
    """log of a number"""
    print("CALLED: log(LogInput) -> LogOutput")
    return LogOutput(result=math.log(input.a))

@mcp.tool()
def strings_to_chars_to_int(input: StringsToIntsInput) -> StringsToIntsOutput:
    """Return the ASCII values of the characters in a word"""
    print("CALLED: strings_to_chars_to_int(StringsToIntsInput) -> StringsToIntsOutput")
    ascii_values = [ord(char) for char in input.string]
    return StringsToIntsOutput(ascii_values=ascii_values)

@mcp.tool()
def int_list_to_exponential_sum(input: ExpSumInput) -> ExpSumOutput:
    """Return sum of exponentials of numbers in a list"""
    print("CALLED: int_list_to_exponential_sum(ExpSumInput) -> ExpSumOutput")
    result = sum(math.exp(i) for i in input.int_list)
    return ExpSumOutput(result=result)

@mcp.tool()
def fibonacci_numbers(input: FibonacciInput) -> FibonacciOutput:
    """Return the first n Fibonacci Numbers"""
    print("CALLED: fibonacci_numbers(FibonacciInput) -> FibonacciOutput")
    if input.n <= 0:
        return []
    fib_sequence = [0, 1]
    for _ in range(2, input.n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return FibonacciOutput(result=fib_sequence[:input.n])

# remainder tool
@mcp.tool()
def remainder(input: RemainderInput) -> RemainderOutput:
    """remainder of two numbers divison"""
    print("CALLED: remainder(RemainderInput) -> RemainderOutput")
    return RemainderOutput(result=input.a % input.b)

# sin tool
@mcp.tool()
def sin(input: SinInput) -> SinOutput:
    """sin of a number"""
    print("CALLED: sin(SinInput) -> SinOutput")
    return SinOutput(result=math.sin(input.a))

# cos tool
@mcp.tool()
def cos(input: CosInput) -> CosOutput:
    """cos of a number"""
    print("CALLED: cos(CosInput) -> CosOutput")
    return CosOutput(result=math.cos(input.a))

# tan tool
@mcp.tool()
def tan(input: TanInput) -> TanOutput:
    """tan of a number"""
    print("CALLED: tan(TanInput) -> TanOutput")
    return TanOutput(result=math.tan(input.a))

# mine tool
@mcp.tool()
def mine(input: MineInput) -> MineOutput:
    """special mining tool"""
    print("CALLED: mine(MineInput) -> MineOutput")
    return MineOutput(result=input.a - input.b - input.b)

class WebpageData(BaseModel):
    url: str
    content: str
    title: str

    # class Config:
    #     from_attributes = True

def process_webpage(url: str, content: str, title: str) -> bool:
    """Process webpage content and update FAISS index."""
    mcp_log("INFO", f"Processing webpage: {url}")
    ROOT = Path(__file__).parent.resolve()
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "webpage_cache.json"

    def content_hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    try:
        # Load existing cache and metadata
        CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
        index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None

        # Check if webpage content has changed
        content_hash_value = content_hash(content)
        if url in CACHE_META and CACHE_META[url] == content_hash_value:
            mcp_log("SKIP", f"Skipping unchanged webpage: {url}")
            return True

        # Process webpage content
        mcp_log("PROC", f"Processing: {url}")
        try:
            # Convert content to markdown if needed
            # converter = MarkItDown()
            # result = converter.convert(str(content))
            # markdown = result.text_content
            chunks = list(chunk_text(content))
            embeddings_for_page = []
            new_metadata = []

            # Generate embeddings for each chunk
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {url}")):
                embedding = get_embedding(chunk)
                embeddings_for_page.append(embedding)
                new_metadata.append({
                    "url": url,
                    "title": title,
                    "chunk": chunk,
                    "chunk_id": f"{url}_{i}",
                        "timestamp": datetime.now().isoformat()
                    })

                # Update index and metadata
                if embeddings_for_page:
                    if index is None:
                        dim = len(embeddings_for_page[0])
                        index = faiss.IndexFlatL2(dim)
                    index.add(np.stack(embeddings_for_page))
                    metadata.extend(new_metadata)
                    CACHE_META[url] = content_hash_value
        except Exception as e:
            mcp_log("ERROR", f"Failed to process {url}: {e}")
            return False

        # Save updated index and metadata
        CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
        METADATA_FILE.write_text(json.dumps(metadata, indent=2))
        if index and index.ntotal > 0:
            faiss.write_index(index, str(INDEX_FILE))
            mcp_log("SUCCESS", f"Saved FAISS index and metadata for {url}")
            return True

        mcp_log("WARN", f"No content to process for {url}")
        return False

    except Exception as e:
        mcp_log("ERROR", f"Failed to process {url}: {e}")
        return False

@mcp.tool()
def process_webpage_tool(data: WebpageData) -> bool:
    """Process a webpage and add to index.
    
    Args:
        data: WebpageData containing url, content, and title
    """
    try:
        process_webpage(data.url, data.content, data.title)
        return True
    except Exception as e:
        mcp_log("ERROR", f"Failed to process webpage: {e}")
        return False

if __name__ == "__main__":
    print("STARTING THE SERVER AT AMAZING LOCATION")
    
    # Create index directory if it doesn't exist
    os.makedirs('faiss_index', exist_ok=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run() # Run without transport for dev server
    else:
        # Start the server in a separate thread
        import threading
        server_thread = threading.Thread(target=lambda: mcp.run(transport="stdio"))
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")