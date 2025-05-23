from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class BboxCoordinates(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BboxCoordinates

class ImageSize(BaseModel):
    width: int
    height: int

class DetectionResponse(BaseModel):
    detections: List[Detection]
    total_objects: int
    result_image: str
    image_size: Optional[ImageSize] = None

class Base64ImageRequest(BaseModel):
    image: str
    confidence: Optional[float] = 0.5

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
