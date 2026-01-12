# ============================================================================
# SEARCH ENDPOINTS (MAIN RECOMMENDATION LOGIC)
# ============================================================================
import numpy as np
import asyncio
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from ..schema.schema import (UserCreate, UserUpdate,
                             ProductCreate, ProductUpdate,
                             SearchRequest, SearchResponse)
from ..utils.embedding_model import embedding_model
from ..core.config import settings
from ..utils.tower_manager import tower_manager

# Create a thread pool executor for running sync operations
executor = ThreadPoolExecutor(max_workers=4)

async def generate_embedding_async(text: str) -> np.ndarray:
    """Generate embedding asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: embedding_model.encode_single(text)
    )

def combine_embeddings(query_emb: np.ndarray, user_emb: np.ndarray, 
                      query_weight: float) -> np.ndarray:
    """
    Combine query and user embeddings with weights
    Formula: search = query*query_weight + user*(1-query_weight)
    """
    combined = (query_weight * query_emb) + ((1 - query_weight) * user_emb)
    # Normalize the combined embedding
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined



async def calculate_similarities(result: dict, user_emb: np.ndarray, tower) -> dict:
    """Calculate cosine similarity between user and product embeddings"""
    # Get product embedding from tower by ID
    product_data = await tower.get_by_id(result['id'])
    
    if product_data and 'embedding' in product_data:
        product_emb = np.array(product_data['embedding'])
        denom = np.linalg.norm(product_emb) * np.linalg.norm(user_emb)
        similarity = round((float(np.dot(product_emb, user_emb) / denom) if denom else 0.0), 4)
    else:
        similarity = 0.00
    
    result["similarity_according_to_user_profile"] = similarity * 100
    return result

router = APIRouter()


@router.post("/search/products", tags=["Search"], response_model=SearchResponse)
async def search_products(search_req: SearchRequest):
    """
    Search products using hybrid query + user preference
    Algorithm: search_embedding = query*query_weight + user*(1-query_weight)
    """
    start = time.time()
    
    # Step 1: Generate query embedding (fast)
    query_task = generate_embedding_async(search_req.query)
    
    # Step 2: Get cached user embedding (very fast)
    user_emb = await tower_manager.get_user_embedding(search_req.user_id)
    if user_emb is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Wait for query embedding
    query_emb = await query_task
    
    # Step 3: Combine embeddings
    search_emb = combine_embeddings(query_emb, user_emb, search_req.query_weight)
    
    # Step 4: Search product tower
    results = await asyncio.gather( 
        *[calculate_similarities(r, user_emb, tower_manager.product_tower) 
        for r in await 
        tower_manager.product_tower.search( query_embedding=search_emb.tolist(), top_k=search_req.top_k )] 
    )
    elapsed = (time.time() - start) * 1000
    
    return SearchResponse(
        results=results,
        search_time_ms=elapsed
    )