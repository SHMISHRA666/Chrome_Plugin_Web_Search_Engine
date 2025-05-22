from perception import PerceptionResult
from models import MemoryItem, SearchResponse, SearchResult, SearchQuery
from memory import MemoryManager
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
import os

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

def generate_plan(
    perception: PerceptionResult,
    memory_items: List[MemoryItem],
    tool_descriptions: Optional[str] = None
) -> str:
    """Generates a plan (tool call) using LLM based on structured perception and memory."""
    
    memory_texts = "\n".join(f"- {m.text}" for m in memory_items) or "None"
    tool_context = f"\nYou have access to the following tools:\n{tool_descriptions}" if tool_descriptions else ""

    prompt = f"""
You are a reasoning-driven AI agent with access to tools. \
Your job is to solve the user's request step-by-step by reasoning\
through the problem, selecting a tool if needed,\
and continuing until you have found all the relevant context\
to produce the RELEVANT_CONTEXT_FOUND.{tool_context}\
For the RELEVANT_CONTEXT_FOUND, we will get top 1-10 results from the search\
and we will need all the results in the plan.

Always follow this loop:

1. Think step-by-step about the problem.
2. If a tool is needed, respond using the format:
   FUNCTION_CALL: tool_name|param1=value1|param2=value2
3. For math queries, when you have found all the relevant context\
   and can provide a final answer, always respond using:
   NO_TOOL_NEEDED: [your final answer]
   For math queries, always respond with NO_TOOL_NEEDED and the final answer.
4. If you found all the relevant contexts, respond using:
   RELEVANT_CONTEXT_FOUND: [Context 1, Context 2, ...]

Guidelines:
- After you receive a result from ANY math tool (add, subtract, multiply, divide, log, \
sqrt, cbrt, factorial, sin, cos, tan, mine, power, strings_to_chars_to_int, \
int_list_to_exponential_sum, fibonacci_numbers, remainder), \
you MUST always respond with:
NO_TOOL_NEEDED: [the final answer]
and you MUST NEVER NEVER use RELEVANT_CONTEXT_FOUND for math queries or call any other tool. \
For trigonometric functions (sin, cos, tan), always output the final answer after the tool result.
For log, the tool only supports natural logarithm (ln). \
If the user asks for log base 10, call the tool with input.a, \
and then respond with NO_TOOL_NEEDED explaining that only \
natural log is supported and show the result.
Never return answers with RELEVANT_CONTEXT_FOUND for a math query.
- Use RELEVANT_CONTEXT_FOUND when you need to found all the relevant contexts
  Example: For "find information about AI", respond with:
  RELEVANT_CONTEXT_FOUND: [search results about AI]
- Respond using EXACTLY ONE of the formats above per step
- Do NOT include extra text, explanation, or formatting
- Use nested keys (e.g., input.string) and square brackets for lists
- For math functions, use the input.key format always and not use any other format
- For ANY search-related queries or questions about web content,\
ALWAYS use the search_pages tool first
- Use search_pages tool for:
  * Finding information about topics
  * Looking up facts
  * Searching for specific content
  * Any question that might have an answer in indexed web pages
- Always and only use NO_TOOL_NEEDED for:
  * Direct mathematical calculations (after using math tools)
- Use RELEVANT_CONTEXT_FOUND when:
  * When you have a definitive answer from previous tool results but not for math queries
  * You have search results that need to be processed
  * You need to analyze multiple pieces of information
  * You want to highlight specific text segments

Examples:
1. For "What is the capital of France?":
   FUNCTION_CALL: search_pages|query="capital of France"
   [receives a detailed content]
   RELEVANT_CONTEXT_FOUND: [Context 1, Context 2, ...]

2. For "Find information about AI":
   FUNCTION_CALL: search_pages|query="information about AI"
   [receives search results]
   RELEVANT_CONTEXT_FOUND: [Context 1, Context 2, ...]

3. For "What is 2+2?":
   FUNCTION_CALL: add|input.a=2|input.b=2
   [receives result 4]
   NO_TOOL_NEEDED: The sum of 2 and 2 is 4

4. For "Tell me about quantum computing":
   FUNCTION_CALL: search_pages|query="quantum computing"
   [receives search results]
   RELEVANT_CONTEXT_FOUND: [Search results about quantum computing]

5. - Use NO_TOOL_NEEDED when you have the final answer for math functions and no more tools are needed
  Example: For "what is 2+2?", after getting result 4, respond with:
  NO_TOOL_NEEDED: The sum of 2 and 2 is 4
  Example: For "what is 35-28?", after getting result 7 from the subtract tool, respond with:
  NO_TOOL_NEEDED: The result of 35 minus 28 is 7
  Example: For "what is the log of 100?", after getting result 4.605 from the log tool, respond with:
  NO_TOOL_NEEDED: The natural logarithm (ln) of 100 is approximately 4.605.
  Example: For "what is the sine of 90?", after getting result 0.893 from the sin tool, respond with:
  NO_TOOL_NEEDED: The sine of 90 (in radians) is approximately 0.893.
  Example: For "what is the cosine of 90?", after getting result -0.448 from the cos tool, respond with:
  NO_TOOL_NEEDED: The cosine of 90 (in radians) is approximately -0.448.
  Example: For "what is the tangent of 90?", after getting result 1.995 from the tan tool, respond with:
  NO_TOOL_NEEDED: The tangent of 90 (in radians) is approximately 1.995.
  Example: For "what is the log base 10 of 100?", respond with:
  NO_TOOL_NEEDED: The tool only supports natural logarithm (ln). ln(100) ≈ 4.605. \
  For log base 10, please use a calculator or another tool.
  Example: For "what is the cube root of 8?", respond with:
  NO_TOOL_NEEDED: The cube root of 8 is 2.
  Example: For "what is the factorial of 5?", respond with:
  NO_TOOL_NEEDED: The factorial of 5 is 120.

6. For sending inputs to math tools, use the following format and for output alwaysuse NO_TOOL_NEEDED:
   FUNCTION_CALL: add|input.a=5|input.b=3
   NO_TOOL_NEEDED: The sum of 5 and 3 is 8
   FUNCTION_CALL: strings_to_chars_to_int|input.string=INDIA
   NO_TOOL_NEEDED: The ascii values of INDIA are [73, 78, 68, 73, 65]
   FUNCTION_CALL: int_list_to_exponential_sum|input.int_list=[73,78,68,73,65]
   NO_TOOL_NEEDED: The exponential sum of [73,78,68,73,65] is 7.59982224609308e+33

7. User asks: "What's the relationship between Cricket and Sachin Tendulkar"
  FUNCTION_CALL: search_pages|query="relationship between Cricket and Sachin Tendulkar"
  [receives a detailed content]
  RELEVANT_CONTEXT_FOUND: [Context 1, Context 2, ...]

You can reference these relevant memories:
{memory_texts}

Input Summary:
- User input: "{perception.user_input}"
- Intent: {perception.intent}
- Entities: {', '.join(perception.entities)}
- Tool hint: {perception.tool_hint or 'None'}

IMPORTANT:
- 🚫 Do NOT invent tools. Use only the tools listed below.
- 📄 If the question may relate to factual knowledge,\
 use the 'search_pages' tool to look for the answer.
- 🧮 If the question is mathematical or needs calculation, use the appropriate math tool.
- 🤖 If the previous tool output already contains factual information, \
DO NOT search again. Instead, summarize the relevant facts and respond with: \
RELEVANT_CONTEXT_FOUND: [Context 1, Context 2, ...]
- 🧮 For math queries, if the previous tool output already\
contains the final answer, always respond with NO_TOOL_NEEDED and the final answer.
- Only repeat `search_pages` if the last result was irrelevant or empty.
- ❌ Do NOT repeat function calls with the same parameters.
- ❌ Do NOT output unstructured responses.
- 📄 For ANY search-related query, ALWAYS use search_pages first
- 🤖 If search_pages returns results, use RELEVANT_CONTEXT_FOUND
- ❌ Do NOT use NO_TOOL_NEEDED for search queries unless search_pages returns no results
- 🧠 Think before each step. Verify intermediate results mentally before proceeding.
- ✅ You have only 3 attempts. Final attempt must be RELEVANT_CONTEXT_FOUND]
- 💥 If unsure or no tool fits, respond with NO_TOOL_NEEDED: [unknown]

- If the user asks a factual question and the search tool \
  returns "No indexed content available for search." or, \
  if the search tool returns no results or, \
  if the search tool returns results that are not relevant to the question, \
  answer the question directly using your own knowledge \
  and respond with NO_TOOL_NEEDED: [final answer]. \
  Do not call the search tool again.
  Example: For "What is the capital of Australia?", if the search tool returns "No indexed content available for search.", respond with:
  NO_TOOL_NEEDED: The capital of Australia is Canberra.
- ✅ You have only 3 attempts. Final attempt must be RELEVANT_CONTEXT_FOUND
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        raw = response.text.strip()
        log("plan", f"LLM output: {raw}")

        for line in raw.splitlines():
            if line.strip().startswith("FUNCTION_CALL:") or line.strip().startswith("NO_TOOL_NEEDED:")\
            or line.strip().startswith("RELEVANT_CONTEXT_FOUND:"):
                return line.strip()

        return raw.strip()

    except Exception as e:
        log("plan", f"⚠️ Decision generation failed: {e}")
        return "NO_TOOL_NEEDED: [Error in tool selection]"
    
def process_search_query(
    query: str,
    memory: MemoryManager,
    top_k: int = 5,
    plan_result: Optional[str] = None
) -> SearchResponse:
    """Process search query and return ranked results with final answer."""
    
    # First, get initial search results
    search_query = SearchQuery(query=query, top_k=top_k)
    initial_results = memory.search(search_query)
    
    if not initial_results.results:
        return SearchResponse(
            results=[],
            total_matches=0,
            final_answer="No relevant results found. Please try a different search query or ensure content has been indexed."
        )
        
    # Use Gemini to improve result ranking, highlighting, and generate final answer
    prompt = f"""
You are a search result ranking and answer generation system. Given a query, search results, and the initial plan, provide improved ranking and a final answer.

Query: "{query}"
Initial Plan: {plan_result or "None"}

Current results:
{chr(10).join(f"- {r.title} ({r.url}): {r.content[:200]}..." for r in initial_results.results)}

Your task:
1. Score each result's relevance (0-1)
2. Find the best matching text segment
3. Generate a comprehensive final answer based on the most relevant results
4. Return in format:
Result_id:int
Result_score:float
Result_highlight_start:int
Result_highlight_end:int
Result_text_segment:str
Result_id|Result_score|Result_highlight_start|Result_highlight_end|Result_text_segment
Like below:
1|5.0|0|110|Relevant text segment 1
2|4.0|10|30|Relevant text segment 2
3|3.0|30|40|Relevant text segment 3
4|2.0|40|60|Relevant text segment 4
5|1.0|50|70|Relevant text segment 5


After all results, add a line:
FINAL_ANSWER: [your comprehensive answer based on the most relevant results]

Guidelines for final answer:
- Synthesize information from multiple relevant results
- Be concise but comprehensive
- Include key facts and relationships
- If results are irrelevant, state that clearly
- If results are conflicting, acknowledge the conflict
- If no clear answer exists, say so explicitly

Output only the results and final answer, one per line.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Process improved results and extract final answer
        improved_results = []
        final_answer = "No clear answer could be generated from the results."
        
        for line in response.text.strip().split('\n'):
            if not line.strip():
                continue
                
            if line.startswith("FINAL_ANSWER:"):
                final_answer = line.replace("FINAL_ANSWER:", "").strip()
                continue
                
            try:
                parts = line.split('|')
                if len(parts) != 5:
                    continue
                    
                result_id, score, start, end, text = parts
                if not result_id.isdigit():
                    continue
                    
                idx = int(result_id)
                if idx < len(initial_results.results):
                    original = initial_results.results[idx]
                    improved_results.append(SearchResult(
                        url=original.url,
                        title=original.title,
                        content=original.content,
                        score=float(score),
                        highlight_start=int(start),
                        highlight_end=int(end)
                    ))
            except (ValueError, IndexError) as e:
                log("decision", f"Failed to parse result: {e}")
                continue
                
        # If no improved results, use original results
        if not improved_results:
            improved_results = initial_results.results
            final_answer = "Found results but couldn't improve ranking. Here are the original results."
                
        return SearchResponse(
            results=improved_results,
            total_matches=len(improved_results),
            final_answer=final_answer
        )
        
    except Exception as e:
        log("decision", f"Failed to improve results: {e}")
        return SearchResponse(
            results=initial_results.results,
            total_matches=len(initial_results.results),
            final_answer="Error processing search results. Showing original results."
        )