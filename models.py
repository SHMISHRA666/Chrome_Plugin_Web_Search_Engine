from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from typing import Literal

class MemoryItem(BaseModel):
    text: Optional[str] = None
    type: Optional[Literal["preference", "tool_output", "fact", "query", "system"]] = "fact"
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = []
    session_id: Optional[str] = None
    url: str
    title: str
    content: str
    timestamp: str = datetime.now().isoformat()
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    url: str
    title: str
    content: str
    score: float
    highlight_start: int
    highlight_end: int
    text: Optional[str] = None
    type: Optional[Literal["preference", "tool_output", "fact", "query", "system"]] = "fact"
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = []
    session_id: Optional[str] = None

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
    type_filter: Optional[str] = None,
    tag_filter: Optional[List[str]] = None,
    session_filter: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_matches: int
    final_answer: Optional[str] = None

class IndexStats(BaseModel):
    total_pages: int
    total_embeddings: int
    last_updated: str
    index_size_bytes: int 

class AddInput(BaseModel):
    a: int
    b: int

class AddOutput(BaseModel):
    result: int

class SqrtInput(BaseModel):
    a: int

class SqrtOutput(BaseModel):
    result: float

class SubtractInput(BaseModel):
    a: int
    b: int

class SubtractOutput(BaseModel):
    result: int
    
class MultiplyInput(BaseModel):
    a: int
    b: int

class MultiplyOutput(BaseModel):
    result: int

class DivideInput(BaseModel):
    a: int
    b: int

class DivideOutput(BaseModel):
    result: float

class PowerInput(BaseModel):
    a: int
    b: int

class PowerOutput(BaseModel):
    result: int

class CubeRootInput(BaseModel):
    a:int

class CubeRootOutput(BaseModel):
    result: float

class FactorialInput(BaseModel):
    a: int

class FactorialOutput(BaseModel):
    result: int

class LogInput(BaseModel):
    a: int

class LogOutput(BaseModel):
    result: float

class RemainderInput(BaseModel):
    a: int
    b: int

class RemainderOutput(BaseModel):
    result: int

class SinInput(BaseModel):
    a: int

class SinOutput(BaseModel):
    result: float

class CosInput(BaseModel):
    a: int

class CosOutput(BaseModel):
    result: float
    
class TanInput(BaseModel):
    a: int

class TanOutput(BaseModel):
    result: float

class MineInput(BaseModel):
    a: int
    b: int

class MineOutput(BaseModel):
    result: int

class StringsToIntsInput(BaseModel):
    string: str

class StringsToIntsOutput(BaseModel):
    ascii_values: List[int]

class ExpSumInput(BaseModel):
    int_list: List[int]

class ExpSumOutput(BaseModel):
    result: float

class FibonacciInput(BaseModel):
    n: int

class FibonacciOutput(BaseModel):
    result: List[int]
