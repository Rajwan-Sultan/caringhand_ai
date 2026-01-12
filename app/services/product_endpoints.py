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


from concurrent.futures import ThreadPoolExecutor
# Create a thread pool executor for running sync operations
executor = ThreadPoolExecutor(max_workers=4)

async def generate_embedding_async(text: str) -> np.ndarray:
    """Generate embedding asynchronously"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: embedding_model.encode_single(text)
    )





router = APIRouter()
# ============================================================================
# PRODUCT ENDPOINTS
# ============================================================================

@router.post("/products/", tags=["Products"])
async def create_product(product: ProductCreate):
    """Create a new product with embedding"""
    start = time.time()
    
    # Generate embedding from name + description + category
    text = f"{product.name}. {product.description}. Category: {product.category}"
    embedding = await generate_embedding_async(text)
    
    # Store in product tower
    metadata = {
        "name": product.name,
        "description": product.description,
        "category": product.category,
        **product.metadata
        
    }
    
    await tower_manager.product_tower.add_items(
        ids=[product.product_id],
        embeddings=[embedding.tolist()],
        metadatas=[metadata]
    )
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "product_id": product.product_id,
        "status": "created",
        "embedding_dim": len(embedding),
        "time_ms": elapsed
    }

@router.put("/products/{product_id}", tags=["Products"])
async def update_product(product_id: str, product_update: ProductUpdate):
    """Update product information and regenerate embedding"""
    start = time.time()
    
    # Get existing product
    existing = await tower_manager.product_tower.get_by_id(product_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Update metadata
    metadata = existing['metadata']
    if product_update.name:
        metadata['name'] = product_update.name
    if product_update.description:
        metadata['description'] = product_update.description
    if product_update.category:
        metadata['category'] = product_update.category
    if product_update.metadata:
        metadata.update(product_update.metadata)
    
    # Regenerate embedding
    text = f"{metadata['name']}. {metadata['description']}. Category: {metadata['category']}"
    embedding = await generate_embedding_async(text)
    
    # Update in tower
    await tower_manager.product_tower.update_item(
        id=product_id,
        embedding=embedding.tolist(),
        metadata=metadata
    )
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "product_id": product_id,
        "status": "updated",
        "time_ms": elapsed
    }

@router.get("/products/{product_id}", tags=["Products"])
async def get_product(product_id: str):
    """Get product information"""
    result = await tower_manager.product_tower.get_by_id(product_id)
    if not result:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {
        "product_id": result['id'],
        "metadata": result['metadata'],
        "embedding_dim": len(result['embedding'])
    }

@router.delete("/products/{product_id}", tags=["Products"])
async def delete_product(product_id: str):
    """Delete a product"""
    await tower_manager.product_tower.delete_item(product_id)
    return {"product_id": product_id, "status": "deleted"}