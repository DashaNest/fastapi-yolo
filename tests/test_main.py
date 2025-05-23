from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import base64
import io
import numpy as np
import torch
from PIL import Image

from app.main import app

# Мокаем YOLO глобально для всех тестов
@patch('app.main.YOLO')
class TestAPI:
    
    def setup_method(self, mock_yolo):
        """Настройка для каждого теста"""
        # Создаем мок модели
        self.mock_model = Mock()
        mock_yolo.return_value = self.mock_model
        
    def test_root_endpoint(self, mock_yolo):
        """Тест корневого эндпоинта"""
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert "YOLO Object Detection API работает" in response.json()["message"]

    def test_health_check(self, mock_yolo):
        """Тест проверки здоровья"""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_detect_invalid_file(self, mock_yolo):
        """Тест с неправильным типом файла"""
        with TestClient(app) as client:
            text_content = b"This is not an image"
            files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
            
            response = client.post("/detect", files=files)
            assert response.status_code == 400
            assert "изображением" in response.json()["detail"]

    def test_detect_base64_missing_image(self, mock_yolo):
        """Тест без поля image"""
        with TestClient(app) as client:
            payload = {"confidence": 0.5}
            response = client.post("/detect_base64", json=payload)
            assert response.status_code == 400
            assert "обязательно" in response.json()["detail"]
