from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from google import genai
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Optional: import log from agent if shared, else define locally
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class PerceptionResult(BaseModel):
    url: str
    title: str
    content: str
    is_indexable: bool
    skip_reason: Optional[str] = None

def is_chrome_url(url: str) -> bool:
    """Check if URL is a Chrome internal URL."""
    return url.startswith('chrome://') or url.startswith('chrome-extension://')

def is_private_url(url: str) -> bool:
    """Check if URL is private/confidential."""
    if is_chrome_url(url):
        return True
        
    private_domains = [
        'gmail.com', 'google.com/mail',
        'whatsapp.com', 'web.whatsapp.com',
        'facebook.com/messages',
        'linkedin.com/messaging',
        'outlook.com', 'office.com',
        'slack.com',
        'teams.microsoft.com'
    ]
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    
    return any(private in f"{domain}{path}" for private in private_domains)

def extract_content(url: str) -> PerceptionResult:
    """Extracts and processes webpage content."""
    try:
        if is_chrome_url(url):
            return PerceptionResult(
                url=url,
                title="Chrome Internal Page",
                content="",
                is_indexable=False,
                skip_reason="Chrome internal URL"
            )
            
        if is_private_url(url):
            return PerceptionResult(
                url=url,
                title="",
                content="",
                is_indexable=False,
                skip_reason="Private/confidential website"
            )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get title
        title = soup.title.string if soup.title else ""
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)
        
        return PerceptionResult(
            url=url,
            title=title,
            content=text,
            is_indexable=True
        )

    except Exception as e:
        log("perception", f"⚠️ Content extraction failed: {e}")
        return PerceptionResult(
            url=url,
            title="",
            content="",
            is_indexable=False,
            skip_reason=str(e)
        )

class PerceptionResultLLM(BaseModel):
    user_input: str
    intent: Optional[str]
    entities: List[str] = []
    tool_hint: Optional[str] = None
    search_type: Optional[str] = None  # e.g., "content", "title", "url"
    filters: Optional[dict] = None  # e.g., {"date_range": "last_week", "domain": "example.com"}


def extract_perception(user_input: str) -> PerceptionResultLLM:
    """Extracts intent, entities, and tool hints using LLM for webpage-related queries"""

    prompt = f"""
You are an AI that extracts structured facts from user input \
related to webpage searching and analysis or related to \
any math query or related to any text based queries.

Input: "{user_input}"

Use the tools given in mcp_server.py to extract the information.
The tools are:
- search_pages
- extract
- add
- sqrt
- subtract
- multiply
- divide
- power
- cbrt
- factorial
- log
- strings_to_chars_to_int
- int_list_to_exponential_sum
- fibonacci_numbers
- remainder
- sin
- cos
- tan
- mine
- process_webpage_tool

Return the response as a Python dictionary with keys:
- intent: (brief phrase about what the user wants to find or analyze on webpages)
- entities: a list of strings representing keywords, URLs, or values to search for
- tool_hint: (name of the MCP tool that might be useful, if any)
- search_type: (type of search - "content", "title", "url", or None if unspecified)
- filters: (dictionary of search filters like date_range, domain, etc. or None if no filters)

Example outputs:
1. For "Find articles about AI from last week on techcrunch.com":
{{"intent": "search for AI articles", \
"entities": ["AI", "articles"], "tool_hint": "search_pages", \
"search_type": "content", "filters": {{"date_range": "last_week", "domain": "techcrunch.com"}}}}

2. For "What's on the homepage of example.com":
{{"intent": "analyze homepage content", \
"entities": ["example.com"], "tool_hint": "search_pages", \
"search_type": "url", "filters": None}}

3. For Find the ASCII values of characters in INDIA and then return sum of exponentials of those values.
{{"intent": "find ASCII values of characters in INDIA and return sum of exponentials of those values", \
"entities": ["INDIA"], "tool_hint": "strings_to_chars_to_int", \
"search_type": None, "filters": None}}

4. For "What is the capital of India?"
{{"intent": "find the capital of India", "entities": ["India"], \
"tool_hint": None, "search_type": "content", "filters": None}}  

For answer returned by LLM like example 4, no tool_hint is \
required as LLM can directly answer the question.

Output only the dictionary on a single line. \
Do NOT wrap it in ```json or other formatting. \
Ensure `entities` is a list of strings, not a dictionary.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        log("perception", f"LLM output: {raw}")

        # Strip Markdown backticks if present
        clean = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        try:
            parsed = eval(clean)
        except Exception as e:
            log("perception", f"⚠️ Failed to parse cleaned output: {e}")
            raise

        # Fix common issues
        if isinstance(parsed.get("entities"), dict):
            parsed["entities"] = list(parsed["entities"].values())

        # Ensure filters is a dictionary or None
        if "filters" in parsed and not isinstance(parsed["filters"], (dict, type(None))):
            parsed["filters"] = None

        return PerceptionResultLLM(user_input=user_input, **parsed)

    except Exception as e:
        log("perception", f"⚠️ Extraction failed: {e}")
        return PerceptionResultLLM(user_input=user_input) 