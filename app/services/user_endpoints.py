
# ============================================================================
# USER ENDPOINTS
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


@router.post("/users/", tags=["Users"])
async def create_user(user: UserCreate):
    """Create a new user with embedding"""
    start = time.time()
    
    # Generate embedding from name + preferences
    text = f"{user.name}. Preferences: {user.preferences}"
    embedding = await generate_embedding_async(text)
    
    # Store in user tower
    metadata = {
        "name": user.name,
        "preferences": user.preferences,
        **user.metadata
    }
    
    await tower_manager.user_tower.add_items(
        ids=[user.user_id],
        embeddings=[embedding.tolist()],
        metadatas=[metadata]
    )
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "user_id": user.user_id,
        "status": "created",
        "embedding_dim": len(embedding),
        "time_ms": elapsed
    }

@router.put("/users/{user_id}", tags=["Users"])
async def update_user(user_id: str, user_update: UserUpdate):
    """Update user information and regenerate embedding"""
    start = time.time()
    
    # Get existing user
    existing = await tower_manager.user_tower.get_by_id(user_id)
    if not existing:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update metadata
    metadata = existing['metadata']
    if user_update.name:
        metadata['name'] = user_update.name
    if user_update.preferences:
        metadata['preferences'] = user_update.preferences
    if user_update.metadata:
        metadata.update(user_update.metadata)
    
    # Regenerate embedding
    text = f"{metadata['name']}. Preferences: {metadata['preferences']}"
    embedding = await generate_embedding_async(text)
    
    # Update in tower
    await tower_manager.user_tower.update_item(
        id=user_id,
        embedding=embedding.tolist(),
        metadata=metadata
    )
    
    # Invalidate cache
    tower_manager.invalidate_user_cache(user_id)
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "user_id": user_id,
        "status": "updated",
        "time_ms": elapsed
    }

@router.get("/users/{user_id}", tags=["Users"])
async def get_user(user_id: str):
    """Get user information"""
    result = await tower_manager.user_tower.get_by_id(user_id)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": result['id'],
        "metadata": result['metadata'],
        "embedding_dim": len(result['embedding'])
    }

@router.delete("/users/{user_id}", tags=["Users"])
async def delete_user(user_id: str):
    """Delete a user"""
    await tower_manager.user_tower.delete_item(user_id)
    tower_manager.invalidate_user_cache(user_id)
    return {"user_id": user_id, "status": "deleted"}