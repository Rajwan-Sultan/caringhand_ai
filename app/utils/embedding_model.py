from ..core.config import Settings
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
class EmbeddingModel:
    _instance = None
    _model = None
    _config = Settings()
    _EMBEDDING_MODEL_NAME = _config.EMBEDDING_MODEL_NAME
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model = SentenceTransformer(cls._EMBEDDING_MODEL_NAME)
            # Optimize for speed
            cls._model.max_seq_length = 128  # Limit sequence length for speed
        return cls._instance
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Fast batch encoding"""
        return self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better similarity
        )
    
    def encode_single(self, text: str) -> np.ndarray:
        """Fast single text encoding"""
        return self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

embedding_model = EmbeddingModel()