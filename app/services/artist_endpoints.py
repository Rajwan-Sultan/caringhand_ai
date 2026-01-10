
# ============================================================================
# ARTIST ENDPOINTS
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
@router.post("/artists/", tags=["Artists"])
async def create_artist(artist: ArtistCreate):
    """Create a new artist with embedding"""
    start = time.time()
    
    # Generate embedding from name + bio + genre
    text = f"{artist.name}. {artist.bio}. Genre: {artist.genre}"
    embedding = await generate_embedding_async(text)
    
    # Store in artist tower
    metadata = {
        "name": artist.name,
        "bio": artist.bio,
        "genre": artist.genre,
        **artist.metadata
    }
    
    await tower_manager.artist_tower.add_items(
        ids=[artist.artist_id],
        embeddings=[embedding.tolist()],
        metadatas=[metadata]
    )
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "artist_id": artist.artist_id,
        "status": "created",
        "embedding_dim": len(embedding),
        "time_ms": elapsed
    }

@router.put("/artists/{artist_id}", tags=["Artists"])
async def update_artist(artist_id: str, artist_update: ArtistUpdate):
    """Update artist information and regenerate embedding"""
    start = time.time()
    
    # Get existing artist
    existing = await tower_manager.artist_tower.get_by_id(artist_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Artist not found")
    
    # Update metadata
    metadata = existing['metadata']
    if artist_update.name:
        metadata['name'] = artist_update.name
    if artist_update.bio:
        metadata['bio'] = artist_update.bio
    if artist_update.genre:
        metadata['genre'] = artist_update.genre
    if artist_update.metadata:
        metadata.update(artist_update.metadata)
    
    # Regenerate embedding
    text = f"{metadata['name']}. {metadata['bio']}. Genre: {metadata['genre']}"
    embedding = await generate_embedding_async(text)
    
    # Update in tower
    await tower_manager.artist_tower.update_item(
        id=artist_id,
        embedding=embedding.tolist(),
        metadata=metadata
    )
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "artist_id": artist_id,
        "status": "updated",
        "time_ms": elapsed
    }

@router.get("/artists/{artist_id}", tags=["Artists"])
async def get_artist(artist_id: str):
    """Get artist information"""
    result = await tower_manager.artist_tower.get_by_id(artist_id)
    if not result:
        raise HTTPException(status_code=404, detail="Artist not found")
    
    return {
        "artist_id": result['id'],
        "metadata": result['metadata'],
        "embedding_dim": len(result['embedding'])
    }

@router.delete("/artists/{artist_id}", tags=["Artists"])
async def delete_artist(artist_id: str):
    """Delete an artist"""
    await tower_manager.artist_tower.delete_item(artist_id)
    return {"artist_id": artist_id, "status": "deleted"}