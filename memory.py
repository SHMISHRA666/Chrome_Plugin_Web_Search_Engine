import numpy as np
import faiss
import requests
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel
from datetime import datetime
import json
import os
from models import MemoryItem, SearchResult, SearchQuery, SearchResponse, IndexStats

class MemoryManager:
    def __init__(self, 
                 embedding_model_url="http://localhost:11434/api/embeddings",
                 model_name="nomic-embed-text",
                 index_path="faiss_index"):
        self.embedding_model_url = embedding_model_url
        self.model_name = model_name
        self.index_path = index_path
        self.index = None
        self.data: List[MemoryItem] = []
        self.embeddings: List[np.ndarray] = []
        

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Nomic."""
        response = requests.post(
            self.embedding_model_url,
            json={"model": self.model_name, "prompt": text}
        )
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)

    def add(self, item: MemoryItem):
        """Add webpage content to index."""
        if not item.content:
            return
            
        emb = self._get_embedding(item.content)
        item.embedding = emb.tolist()
        
        self.embeddings.append(emb)
        self.data.append(item)

        # Initialize or add to index
        if self.index is None:
            self.index = faiss.IndexFlatL2(len(emb))
        self.index.add(np.stack([emb]))

    def search(self, query: SearchQuery) -> SearchResponse:
        """Search for content in indexed pages."""
        if not self.index or len(self.data) == 0:
            return SearchResponse(results=[], total_matches=0)

        query_vec = self._get_embedding(query.query).reshape(1, -1)
        D, I = self.index.search(query_vec, query.top_k*2)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= len(self.data):
                continue
                
            item = self.data[idx]
            content = item.content
            
            # Find the best matching text segment with context
            highlight_start, highlight_end = self._find_best_match(content, query.query)
            
            results.append(SearchResult(
                url=item.url,
                title=item.title,
                content=content,
                score=float(score),
                highlight_start=highlight_start,
                highlight_end=highlight_end
            ))

        return SearchResponse(
            results=results,
            total_matches=len(results)
        )

    def _find_best_match(self, content: str, query: str) -> tuple[int, int]:
        """Find the best matching text segment with context.
        
        Args:
            content: The full text content to search in
            query: The search query
            
        Returns:
            Tuple of (highlight_start, highlight_end) indices
        """
        # Split query into words and clean
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        if not query_words:
            return 0, 0
            
        content_lower = content.lower()
        best_score = -1
        best_start = 0
        best_end = 0
        
        # Try different window sizes to find the best context
        window_sizes = [100, 200, 300]  # Characters to look at
        
        for window_size in window_sizes:
            # Slide window through content
            for i in range(0, len(content), window_size // 2):
                window = content_lower[i:i + window_size]
                
                # Count query word matches in this window
                matches = sum(1 for word in query_words if word in window)
                if matches == 0:
                    continue
                    
                # Calculate score based on:
                # 1. Number of matching words
                # 2. Proximity of matches
                # 3. Position of matches in window
                score = matches * 2
                
                # Find the actual positions of matches
                match_positions = []
                for word in query_words:
                    pos = window.find(word)
                    if pos != -1:
                        match_positions.append(pos)
                
                if match_positions:
                    # Add proximity bonus
                    proximity = window_size - (max(match_positions) - min(match_positions))
                    score += proximity / window_size
                    
                    # Add position bonus (prefer matches near start of window)
                    position_bonus = 1 - (min(match_positions) / window_size)
                    score += position_bonus
                
                if score > best_score:
                    best_score = score
                    # Find the actual text boundaries
                    first_match = min(match_positions)
                    last_match = max(match_positions) + len(query_words[-1])
                    
                    # Add some context before and after
                    context_before = 20
                    context_after = 20
                    
                    best_start = i + max(0, first_match - context_before)
                    best_end = i + min(len(content), last_match + context_after)
        
        return best_start, best_end

    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        return IndexStats(
            total_pages=len(self.data),
            total_embeddings=len(self.embeddings),
            last_updated=datetime.now().isoformat(),
            index_size_bytes=os.path.getsize(os.path.join(self.index_path, "index.bin")) if self.index else 0
        ) 
    
    def bulk_add(self, items: List[MemoryItem]):
        for item in items:
            self.add(item)