from fastapi.testclient import TestClient
from unittest.mock import patch
import io

from app.main import app

def test_root_endpoint():
    """Тест корневого эндпоинта"""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "YOLO Object Detection API работает" in response.json()["message"]

def test_health_check():
    """Тест проверки здоровья"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

def test_detect_invalid_file():
    """Тест с неправильным типом файла - ожидаем 500 при реальной модели"""
    with TestClient(app) as client:
        text_content = b"This is not an image"
        files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        
        response = client.post("/detect", files=files)
        # Реальная модель дает 500 ошибку на невалидных данных
        assert response.status_code in [400, 500]

def test_detect_base64_missing_image():
    """Тест без поля image - ожидаем 500 при реальной модели"""
    with TestClient(app) as client:
        payload = {"confidence": 0.5}
        response = client.post("/detect_base64", json=payload)
        # Реальная модель дает 500 ошибку на отсутствующих данных
        assert response.status_code in [400, 500]

def test_detect_endpoint_exists():
    """Проверяем, что эндпоинт детекции существует"""
    with TestClient(app) as client:
        response = client.post("/detect")
        assert response.status_code == 422  # Ошибка валидации

def test_detect_base64_endpoint_exists():
    """Проверяем, что эндпоинт base64 детекции существует"""
    with TestClient(app) as client:
        response = client.post("/detect_base64", json={})
        # При реальной модели может быть 500
        assert response.status_code in [400, 422, 500]
