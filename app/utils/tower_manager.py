from .vectordb_adapter import ChromaDBAdapter
from typing import List, Dict, Optional
import numpy as np
class TowerManager:
    """Manages the towers with caching"""
    
    def __init__(self):
        # Initialize n separate towers (collections)
        self.user_tower = ChromaDBAdapter("user_tower")
        self.artist_tower = ChromaDBAdapter("artist_tower")
        self.product_tower = ChromaDBAdapter("product_tower")
        # self.song_tower = ChromaDBAdapter("song_tower")
        
        # Cache for user embeddings (most frequently accessed)
        self._user_embedding_cache: Dict[str, np.ndarray] = {}
        
    async def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get user embedding with caching"""
        if user_id in self._user_embedding_cache:
            return self._user_embedding_cache[user_id]
        
        result = await self.user_tower.get_by_id(user_id)
        if result:
            embedding = np.array(result['embedding'])
            self._user_embedding_cache[user_id] = embedding
            return embedding
        return None
    
    def invalidate_user_cache(self, user_id: str):
        """Invalidate cache when user is updated"""
        if user_id in self._user_embedding_cache:
            del self._user_embedding_cache[user_id]

tower_manager = TowerManager()