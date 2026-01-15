# ============================================================================
# Top 5 Recommended Jobs Service
# ============================================================================
import numpy as np
import asyncio
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

from ..schema.schema import SearchResponse
from ..utils.embedding_model import embedding_model
from ..core.config import settings
from ..utils.tower_manager import tower_manager

executor = ThreadPoolExecutor(max_workers=4)
router = APIRouter()




async def calculate_similarities(result: dict, user_emb: np.ndarray, tower) -> dict:
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


@router.get("/top_products", tags=["Top 5 Job"], response_model=SearchResponse)
async def search_products(user_id: str,top_k: int = 5):

    
    user_emb = await tower_manager.get_user_embedding(user_id)
    if user_emb is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    
    results = await asyncio.gather( 
        *[calculate_similarities(r, user_emb, tower_manager.product_tower) 
        for r in await 
        tower_manager.product_tower.search( query_embedding=user_emb.tolist(), top_k=top_k)] 
    )
    
    return SearchResponse(
        results=results,
    )