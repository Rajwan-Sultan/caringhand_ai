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
                             ArtistCreate, ArtistUpdate,
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
    results = await tower_manager.product_tower.search(
        query_embedding=search_emb.tolist(),
        top_k=search_req.top_k
    )
    
    # Add similarity score to each result using list comprehension
    results = [{**r, "similarity_according_to_search_and_account": round(max(0, min(100, (1 - r["distance"]) * 100)), 2)} for r in results]
    
    elapsed = (time.time() - start) * 1000
    
    return SearchResponse(
        results=results,
        search_time_ms=elapsed
    )

@router.post("/search/artists", tags=["Search"], response_model=SearchResponse)
async def search_artists(search_req: SearchRequest):
    """
    Search artists using hybrid query + user preference
    """
    start = time.time()
    
    # Generate query embedding and get user embedding in parallel
    query_task = generate_embedding_async(search_req.query)
    user_emb = await tower_manager.get_user_embedding(search_req.user_id)
    
    if user_emb is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    query_emb = await query_task
    
    # Combine embeddings
    search_emb = combine_embeddings(query_emb, user_emb, search_req.query_weight)
    
    # Search artist tower
    results = await tower_manager.artist_tower.search(
        query_embedding=search_emb.tolist(),
        top_k=search_req.top_k
    )
    
    results = [{**r, "similarity_according_to_search_and_account": round(max(0, min(100, (1 - r["distance"]) * 100)), 2)} for r in results]
    elapsed = (time.time() - start) * 1000

    return SearchResponse(
        results=results,
        search_time_ms=elapsed
    )

@router.post("/search/combined", tags=["Search"])
async def search_combined(search_req: SearchRequest):
    """
    Search both products and artists, return combined ranked results
    """
    start = time.time()
    
    # Generate embeddings once
    query_task = generate_embedding_async(search_req.query)
    user_emb = await tower_manager.get_user_embedding(search_req.user_id)
    
    if user_emb is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    query_emb = await query_task
    search_emb = combine_embeddings(query_emb, user_emb, search_req.query_weight)
    
    # Search both towers in parallel
    products_task = tower_manager.product_tower.search(
        query_embedding=search_emb.tolist(),
        top_k=search_req.top_k
    )
    artists_task = tower_manager.artist_tower.search(
        query_embedding=search_emb.tolist(),
        top_k=search_req.top_k
    )
    
    products, artists = await asyncio.gather(products_task, artists_task)
    
    # Add type field
    for p in products:
        p['type'] = 'product'
    for a in artists:
        a['type'] = 'artist'
    
    # Combine and sort by distance
    combined = products + artists
    combined.sort(key=lambda x: x['distance'])
    
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "results": combined[:search_req.top_k],
        "search_time_ms": elapsed,
        "products_count": len(products),
        "artists_count": len(artists)
    }