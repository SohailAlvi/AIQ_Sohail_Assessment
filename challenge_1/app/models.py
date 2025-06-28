from pydantic import BaseModel
from typing import Optional, List, Tuple


class DetectionResult(BaseModel):
    id: int
    ref_id: str                      # 👈 Unique ID per detected object
    score: float
    bbox: List[int]
    centroid: Optional[Tuple[int, int]]
    radius: Optional[float]


class InferenceResponse(BaseModel):
    object_count: int
    objects: List[DetectionResult]
    visualization_url: Optional[str]  # 👈 MinIO URL string (not raw bytes)
