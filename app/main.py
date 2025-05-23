from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import base64
import io
from contextlib import asynccontextmanager
import torch
from typing import List, Dict, Any

# Контекстный менеджер для загрузки модели при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загружаем модель при старте приложения
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO('yolov5m.pt').to(device)
    app.state.model = model
    print(f"Модель загружена на устройство: {device}")
    yield
    # Очищаем ресурсы при завершении
    del app.state.model

app = FastAPI(
    title="YOLO Object Detection API",
    description="API для детекции объектов с использованием YOLOv5",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS для работы со Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Проверка работоспособности API"""
    return {"message": "YOLO Object Detection API работает"}

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "model_loaded": hasattr(app.state, 'model')}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Детекция объектов на изображении
    """
    try:
        # Проверяем тип файла
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        # Читаем изображение
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Выполняем детекцию
        results = app.state.model(image, conf=0.5)
        
        # Обрабатываем результаты
        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results[0].names[class_id]
                
                # Координаты bbox
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
        
        # Создаем изображение с результатами
        result_image = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в base64
        img_pil = Image.fromarray(result_image_rgb)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "detections": detections,
            "total_objects": len(detections),
            "result_image": img_base64,
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

@app.post("/detect_base64")
async def detect_objects_base64(image_data: dict):
    """
    Детекция объектов из base64 изображения
    """
    try:
        # Декодируем base64
        image_base64 = image_data.get("image")
        if not image_base64:
            raise HTTPException(status_code=400, detail="Поле 'image' обязательно")
        
        confidence = image_data.get("confidence", 0.5)
        
        # Декодируем изображение
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Выполняем детекцию
        results = app.state.model(image, conf=confidence)
        
        # Обрабатываем результаты (аналогично предыдущему методу)
        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence_val = float(box.conf[0])
                class_name = results[0].names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence_val, 3),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })
        
        # Создаем изображение с результатами
        result_image = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в base64
        img_pil = Image.fromarray(result_image_rgb)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "detections": detections,
            "total_objects": len(detections),
            "result_image": result_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
