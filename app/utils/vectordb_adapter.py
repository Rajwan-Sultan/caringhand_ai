
class VectorDBInterface:
    """
    Abstract interface for vector database operations.
    To replace with another vector DB:
    1. Implement methods: add_items, update_item, delete_item, search, get_by_id
    2. Replace ChromaDBAdapter with your new adapter (WeaviateAdapter, PineconeAdapter, etc.)
    3. Update initialization in TowerManager

    This is to ensure modularity and ease of swapping vector DB backends.
    """
    
    async def add_items(self, ids: List[str], embeddings: List[List[float]], 
                       metadatas: List[Dict]) -> None:
        raise NotImplementedError
    
    async def update_item(self, id: str, embedding: List[float], 
                         metadata: Dict) -> None:
        raise NotImplementedError
    
    async def delete_item(self, id: str) -> None:
        raise NotImplementedError
    
    async def search(self, query_embedding: List[float], 
                    top_k: int = 10, where: Optional[Dict] = None) -> List[Dict]:
        raise NotImplementedError
    
    async def get_by_id(self, id: str) -> Optional[Dict]:
        raise NotImplementedError

class ChromaDBAdapter(VectorDBInterface):
    
    def __init__(self, collection_name: str):
        # Use persistent storage
        persist_directory = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
    
    async def add_items(self, ids: List[str], embeddings: List[List[float]], 
                       metadatas: List[Dict]) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            lambda: self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        )
    
    async def update_item(self, id: str, embedding: List[float], 
                         metadata: Dict) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            lambda: self.collection.update(
                ids=[id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        )
    
    async def delete_item(self, id: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            lambda: self.collection.delete(ids=[id])
        )
    
    async def search(self, query_embedding: List[float], 
                    top_k: int = 10, where: Optional[Dict] = None) -> List[Dict]:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            lambda: self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
        )
         
        # Format results
        formatted = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                })
        return formatted
    
    async def get_by_id(self, id: str) -> Optional[Dict]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: self.collection.get(ids=[id], include=['embeddings', 'metadatas'])
        )
        
        if result['ids']:
            return {
                'id': result['ids'][0],
                'embedding': result['embeddings'][0],
                'metadata': result['metadatas'][0]
            }
        return None

