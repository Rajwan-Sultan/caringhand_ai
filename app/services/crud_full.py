# import numpy as np
# import asyncio
# import time
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel

# from ..schema.schema import (UserCreate, UserUpdate,
#                              ProductCreate, ProductUpdate,
#                              SearchRequest, SearchResponse)
# from ..utils.embedding_model import embedding_model
# from ..core.config import settings
# from ..utils.tower_manager import tower_manager

# router = APIRouter()

# from concurrent.futures import ThreadPoolExecutor
# # Create a thread pool executor for running sync operations
# executor = ThreadPoolExecutor(max_workers=4)

# async def generate_embedding_async(text: str) -> np.ndarray:
#     """Generate embedding asynchronously"""
#     loop = asyncio.get_event_loop()
#     return await loop.run_in_executor(
#         executor,
#         lambda: embedding_model.encode_single(text)
#     )

# def combine_embeddings(query_emb: np.ndarray, user_emb: np.ndarray, 
#                       query_weight: float) -> np.ndarray:
#     """
#     Combine query and user embeddings with weights
#     Formula: search = query*query_weight + user*(1-query_weight)
#     """
#     combined = (query_weight * query_emb) + ((1 - query_weight) * user_emb)
#     # Normalize the combined embedding
#     norm = np.linalg.norm(combined)
#     if norm > 0:
#         combined = combined / norm
#     return combined

# # ============================================================================
# # USER ENDPOINTS
# # ============================================================================

# @router.post("/users/", tags=["Users"])
# async def create_user(user: UserCreate):
#     """Create a new user with embedding"""
#     start = time.time()
    
#     # Generate embedding from name + preferences
#     text = f"{user.name}. Preferences: {user.preferences}"
#     embedding = await generate_embedding_async(text)
    
#     # Store in user tower
#     metadata = {
#         "name": user.name,
#         "preferences": user.preferences,
#         **user.metadata
#     }
    
#     await tower_manager.user_tower.add_items(
#         ids=[user.user_id],
#         embeddings=[embedding.tolist()],
#         metadatas=[metadata]
#     )
    
#     elapsed = (time.time() - start) * 1000
    
#     return {
#         "user_id": user.user_id,
#         "status": "created",
#         "embedding_dim": len(embedding),
#         "time_ms": elapsed
#     }

# @router.put("/users/{user_id}", tags=["Users"])
# async def update_user(user_id: str, user_update: UserUpdate):
#     """Update user information and regenerate embedding"""
#     start = time.time()
    
#     # Get existing user
#     existing = await tower_manager.user_tower.get_by_id(user_id)
#     if not existing:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     # Update metadata
#     metadata = existing['metadata']
#     if user_update.name:
#         metadata['name'] = user_update.name
#     if user_update.preferences:
#         metadata['preferences'] = user_update.preferences
#     if user_update.metadata:
#         metadata.update(user_update.metadata)
    
#     # Regenerate embedding
#     text = f"{metadata['name']}. Preferences: {metadata['preferences']}"
#     embedding = await generate_embedding_async(text)
    
#     # Update in tower
#     await tower_manager.user_tower.update_item(
#         id=user_id,
#         embedding=embedding.tolist(),
#         metadata=metadata
#     )
    
#     # Invalidate cache
#     tower_manager.invalidate_user_cache(user_id)
    
#     elapsed = (time.time() - start) * 1000
    
#     return {
#         "user_id": user_id,
#         "status": "updated",
#         "time_ms": elapsed
#     }

# @router.get("/users/{user_id}", tags=["Users"])
# async def get_user(user_id: str):
#     """Get user information"""
#     result = await tower_manager.user_tower.get_by_id(user_id)
#     if not result:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     return {
#         "user_id": result['id'],
#         "metadata": result['metadata'],
#         "embedding_dim": len(result['embedding'])
#     }

# @router.delete("/users/{user_id}", tags=["Users"])
# async def delete_user(user_id: str):
#     """Delete a user"""
#     await tower_manager.user_tower.delete_item(user_id)
#     tower_manager.invalidate_user_cache(user_id)
#     return {"user_id": user_id, "status": "deleted"}

# # ============================================================================
# # ARTIST ENDPOINTS
# # ============================================================================

# @router.post("/artists/", tags=["Artists"])
# async def create_artist(artist: ArtistCreate):
#     """Create a new artist with embedding"""
#     start = time.time()
    
#     # Generate embedding from name + bio + genre
#     text = f"{artist.name}. {artist.bio}. Genre: {artist.genre}"
#     embedding = await generate_embedding_async(text)
    
#     # Store in artist tower
#     metadata = {
#         "name": artist.name,
#         "bio": artist.bio,
#         "genre": artist.genre,
#         **artist.metadata
#     }
    
#     await tower_manager.artist_tower.add_items(
#         ids=[artist.artist_id],
#         embeddings=[embedding.tolist()],
#         metadatas=[metadata]
#     )
    
#     elapsed = (time.time() - start) * 1000
    
#     return {
#         "artist_id": artist.artist_id,
#         "status": "created",
#         "embedding_dim": len(embedding),
#         "time_ms": elapsed
#     }

# @router.put("/artists/{artist_id}", tags=["Artists"])
# async def update_artist(artist_id: str, artist_update: ArtistUpdate):
#     """Update artist information and regenerate embedding"""
#     start = time.time()
    
#     # Get existing artist
#     existing = await tower_manager.artist_tower.get_by_id(artist_id)
#     if not existing:
#         raise HTTPException(status_code=404, detail="Artist not found")
    
#     # Update metadata
#     metadata = existing['metadata']
#     if artist_update.name:
#         metadata['name'] = artist_update.name
#     if artist_update.bio:
#         metadata['bio'] = artist_update.bio
#     if artist_update.genre:
#         metadata['genre'] = artist_update.genre
#     if artist_update.metadata:
#         metadata.update(artist_update.metadata)
    
#     # Regenerate embedding
#     text = f"{metadata['name']}. {metadata['bio']}. Genre: {metadata['genre']}"
#     embedding = await generate_embedding_async(text)
    
#     # Update in tower
#     await tower_manager.artist_tower.update_item(
#         id=artist_id,
#         embedding=embedding.tolist(),
#         metadata=metadata
#     )
    
#     elapsed = (time.time() - start) * 1000
    
#     return {
#         "artist_id": artist_id,
#         "status": "updated",
#         "time_ms": elapsed
#     }

# @router.get("/artists/{artist_id}", tags=["Artists"])
# async def get_artist(artist_id: str):
#     """Get artist information"""
#     result = await tower_manager.artist_tower.get_by_id(artist_id)
#     if not result:
#         raise HTTPException(status_code=404, detail="Artist not found")
    
#     return {
#         "artist_id": result['id'],
#         "metadata": result['metadata'],
#         "embedding_dim": len(result['embedding'])
#     }

# @router.delete("/artists/{artist_id}", tags=["Artists"])
# async def delete_artist(artist_id: str):
#     """Delete an artist"""
#     await tower_manager.artist_tower.delete_item(artist_id)
#     return {"artist_id": artist_id, "status": "deleted"}

# # ============================================================================
# # PRODUCT ENDPOINTS
# # ============================================================================

# @router.post("/products/", tags=["Products"])
# async def create_product(product: ProductCreate):
#     """Create a new product with embedding"""
#     start = time.time()
    
#     # Generate embedding from name + description + category
#     text = f"{product.name}. {product.description}. Category: {product.category}"
#     embedding = await generate_embedding_async(text)
    
#     # Store in product tower
#     metadata = {
#         "name": product.name,
#         "description": product.description,
#         # "artist_id": product.artist_id,
#         "category": product.category,
#         **product.metadata
        
#     }
    
#     await tower_manager.product_tower.add_items(
#         ids=[product.product_id],
#         embeddings=[embedding.tolist()],
#         metadatas=[metadata]
#     )
    
#     elapsed = (time.time() - start) * 1000
    
#     return {
#         "product_id": product.product_id,
#         "status": "created",
#         "embedding_dim": len(embedding),
#         "time_ms": elapsed
#     }

# @router.put("/products/{product_id}", tags=["Products"])
# async def update_product(product_id: str, product_update: ProductUpdate):
#     """Update product information and regenerate embedding"""
#     start = time.time()
    
#     # Get existing product
#     existing = await tower_manager.product_tower.get_by_id(product_id)
#     if not existing:
#         raise HTTPException(status_code=404, detail="Product not found")
    
#     # Update metadata
#     metadata = existing['metadata']
#     if product_update.name:
#         metadata['name'] = product_update.name
#     if product_update.description:
#         metadata['description'] = product_update.description
#     # if product_update.artist_id:
#     #     metadata['artist_id'] = product_update.artist_id
#     if product_update.category:
#         metadata['category'] = product_update.category
#     if product_update.metadata:
#         metadata.update(product_update.metadata)
    
#     # Regenerate embedding
#     text = f"{metadata['name']}. {metadata['description']}. Category: {metadata['category']}"
#     embedding = await generate_embedding_async(text)
    
#     # Update in tower
#     await tower_manager.product_tower.update_item(
#         id=product_id,
#         embedding=embedding.tolist(),
#         metadata=metadata
#     )
    
#     elapsed = (time.time() - start) * 1000
    
#     return {
#         "product_id": product_id,
#         "status": "updated",
#         "time_ms": elapsed
#     }

# @router.get("/products/{product_id}", tags=["Products"])
# async def get_product(product_id: str):
#     """Get product information"""
#     result = await tower_manager.product_tower.get_by_id(product_id)
#     if not result:
#         raise HTTPException(status_code=404, detail="Product not found")
    
#     return {
#         "product_id": result['id'],
#         "metadata": result['metadata'],
#         "embedding_dim": len(result['embedding'])
#     }

# @router.delete("/products/{product_id}", tags=["Products"])
# async def delete_product(product_id: str):
#     """Delete a product"""
#     await tower_manager.product_tower.delete_item(product_id)
#     return {"product_id": product_id, "status": "deleted"}

# # ============================================================================
# # SEARCH ENDPOINTS (MAIN RECOMMENDATION LOGIC)
# # ============================================================================

# @router.post("/search/products", tags=["Search"], response_model=SearchResponse)
# async def search_products(search_req: SearchRequest):
#     """
#     Search products using hybrid query + user preference
#     Algorithm: search_embedding = query*query_weight + user*(1-query_weight)
#     """
#     start = time.time()
    
#     # Step 1: Generate query embedding (fast)
#     query_task = generate_embedding_async(search_req.query)
    
#     # Step 2: Get cached user embedding (very fast)
#     user_emb = await tower_manager.get_user_embedding(search_req.user_id)
#     if user_emb is None:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     # Wait for query embedding
#     query_emb = await query_task
    
#     # Step 3: Combine embeddings
#     search_emb = combine_embeddings(query_emb, user_emb, search_req.query_weight)
    
#     # Step 4: Search product tower
#     results = await tower_manager.product_tower.search(
#         query_embedding=search_emb.tolist(),
#         top_k=search_req.top_k
#     )
    
#     elapsed = (time.time() - start) * 1000
    
#     return SearchResponse(
#         results=results,
#         search_time_ms=elapsed
#     )

# @router.post("/search/artists", tags=["Search"], response_model=SearchResponse)
# async def search_artists(search_req: SearchRequest):
#     """
#     Search artists using hybrid query + user preference
#     """
#     start = time.time()
    
#     # Generate query embedding and get user embedding in parallel
#     query_task = generate_embedding_async(search_req.query)
#     user_emb = await tower_manager.get_user_embedding(search_req.user_id)
    
#     if user_emb is None:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     query_emb = await query_task
    
#     # Combine embeddings
#     search_emb = combine_embeddings(query_emb, user_emb, search_req.query_weight)
    
#     # Search artist tower
#     results = await tower_manager.artist_tower.search(
#         query_embedding=search_emb.tolist(),
#         top_k=search_req.top_k
#     )
    
#     elapsed = (time.time() - start) * 1000
    
#     return SearchResponse(
#         results=results,
#         search_time_ms=elapsed
#     )

# @router.post("/search/combined", tags=["Search"])
# async def search_combined(search_req: SearchRequest):
#     """
#     Search both products and artists, return combined ranked results
#     """
#     start = time.time()
    
#     # Generate embeddings once
#     query_task = generate_embedding_async(search_req.query)
#     user_emb = await tower_manager.get_user_embedding(search_req.user_id)
    
#     if user_emb is None:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     query_emb = await query_task
#     search_emb = combine_embeddings(query_emb, user_emb, search_req.query_weight)
    
#     # Search both towers in parallel
#     products_task = tower_manager.product_tower.search(
#         query_embedding=search_emb.tolist(),
#         top_k=search_req.top_k
#     )
#     artists_task = tower_manager.artist_tower.search(
#         query_embedding=search_emb.tolist(),
#         top_k=search_req.top_k
#     )
    
#     products, artists = await asyncio.gather(products_task, artists_task)
    
#     # Add type field
#     for p in products:
#         p['type'] = 'product'
#     for a in artists:
#         a['type'] = 'artist'
    
#     # Combine and sort by distance
#     combined = products + artists
#     combined.sort(key=lambda x: x['distance'])
    
#     elapsed = (time.time() - start) * 1000
    
#     return {
#         "results": combined[:search_req.top_k],
#         "search_time_ms": elapsed,
#         "products_count": len(products),
#         "artists_count": len(artists)
#     }

# # ============================================================================
# # HEALTH CHECK
# # ============================================================================

# @router.get("/", tags=["Health"])
# async def root():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "system": "Three Tower Recommendation System",
#         "embedding_model": EMBEDDING_MODEL_NAME,
#         "embedding_dim": 384,
#         "towers": ["user_tower", "artist_tower", "product_tower"]
#     }

# @router.get("/health", tags=["Health"])
# async def health_check():
#     """Detailed health check"""
#     return {
#         "status": "healthy",
#         "embedding_model_loaded": embedding_model._model is not None,
#         "cache_size": len(tower_manager._user_embedding_cache),
#         "query_weight": settings.QUERY_WEIGHT
#     }

# # # ============================================================================
# # # STARTUP
# # # ============================================================================

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run("main:router", host="0.0.0.0", port=8062,reload=True)