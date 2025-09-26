from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance, PayloadSchemaType
import uuid
import time


class ChatHistoryStore:
    """
    Simple Qdrant-backed chat history with a separate collection from vector DB.
    - Uses a tiny 1-dim dummy vector [0.0] for compatibility
    - Filters by session_id; sorts by timestamp client-side
    """

    def __init__(self, collection_name: str, qdrant_url: str, qdrant_api_key: Optional[str] = None):
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
        # Ensure payload index for filtering by session_id
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Index may already exist or server may not support; ignore
            pass

    def add_message(self, session_id: str, role: str, content: str, ts: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "ts": float(ts if ts is not None else time.time()),
        }
        if extra:
            payload.update(extra)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],
            payload=payload,
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def list_messages(self, session_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        # Use scroll to fetch payloads by session_id
        flt = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])
        all_payloads: List[Dict[str, Any]] = []
        next_page = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=min(256, limit - len(all_payloads)) if limit else 256,
                offset=next_page,
            )
            points, next_page = result
            for p in points:
                if p.payload:
                    all_payloads.append(p.payload)
            if not next_page or (limit and len(all_payloads) >= limit):
                break
        all_payloads.sort(key=lambda x: x.get("ts", 0.0))
        return all_payloads[:limit] if limit else all_payloads

    def reset_session(self, session_id: str) -> None:
        flt = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])
        self.client.delete(collection_name=self.collection_name, points_selector=flt)

    def drop_collection(self) -> None:
        """Dangerous: deletes the entire chat history collection."""
        self.client.delete_collection(self.collection_name)


