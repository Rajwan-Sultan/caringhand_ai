
# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "Three Tower Recommendation System",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": 384,
        "towers": ["user_tower", "artist_tower", "product_tower"]
    }

@router.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "embedding_model_loaded": embedding_model._model is not None,
        "cache_size": len(tower_manager._user_embedding_cache),
        "query_weight": settings.QUERY_WEIGHT
    }

# ============================================================================
# STARTUP
# ============================================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:router", host="0.0.0.0", port=8062,reload=True)