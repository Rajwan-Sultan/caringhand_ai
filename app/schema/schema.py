from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ..core.config import settings

class UserCreate(BaseModel):
    user_id: str
    name: str
    preferences: str  # Text description of preferences
    metadata: Optional[Dict[str, Any]] = {}

class UserUpdate(BaseModel):
    name: Optional[str] = None
    preferences: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProductCreate(BaseModel):
    product_id: str
    name: str
    description: str
    category: str
    metadata: Optional[Dict[str, Any]] = {}

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 10
    query_weight: float = settings.QUERY_WEIGHT

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    search_time_ms: float