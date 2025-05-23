from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import base64
import io
import numpy as np
import torch
from PIL import Image

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
        # Мокаем наличие модели
        app.state.model = Mock()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

def test_detect_with_image():
    """Тест детекции с изображением"""
    with TestClient(app) as client:
        # Создаем мок модели
        mock_model = Mock()
        
        # Настраиваем мок box
        mock_box = Mock()
        mock_box.cls = torch.tensor([0])
        mock_box.conf = torch.tensor([0.85])
        mock_box.xyxy = torch.tensor([[100, 100, 200, 200]])
        
        # Настраиваем мок результата
        mock_result = Mock()
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}
        
        # Создаем простое изображение для plot
        plot_result = np.zeros((300, 300, 3), dtype=np.uint8)
        plot_result[:, :] = [255, 0, 0]  # Красное изображение
        mock_result.plot.return_value = plot_result
        
        mock_model.return_value = [mock_result]
        
        # Устанавливаем мок в состояние приложения
        app.state.model = mock_model
        
        # Создаем тестовое изображение
        image = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        # Отправляем запрос
        files = {"file": ("test.jpg", buffer, "image/jpeg")}
        response = client.post("/detect", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "total_objects" in data
        assert "result_image" in data
        assert data["total_objects"] == 1

def test_detect_base64():
    """Тест детекции с base64"""
    with TestClient(app) as client:
        # Создаем мок модели
        mock_model = Mock()
        
        # Настраиваем мок результата без объектов
        mock_result = Mock()
        mock_result.boxes = []  # Нет объектов
        
        # Создаем простое изображение для plot
        plot_result = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_result.plot.return_value = plot_result
        
        mock_model.return_value = [mock_result]
        app.state.model = mock_model
        
        # Создаем base64 изображение
        image = Image.new('RGB', (50, 50), color='blue')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        payload = {"image": image_base64}
        response = client.post("/detect_base64", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_objects"] == 0
        assert "result_image" in data

def test_detect_invalid_file():
    """Тест с неправильным типом файла"""
    with TestClient(app) as client:
        text_content = b"This is not an image"
        files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        
        response = client.post("/detect", files=files)
        assert response.status_code == 400
        assert "изображением" in response.json()["detail"]

def test_detect_base64_missing_image():
    """Тест без поля image"""
    with TestClient(app) as client:
        payload = {"confidence": 0.5}
        response = client.post("/detect_base64", json=payload)
        assert response.status_code == 400
        assert "обязательно" in response.json()["detail"]

def test_detect_with_multiple_objects():
    """Тест детекции с несколькими объектами"""
    with TestClient(app) as client:
        # Создаем мок модели
        mock_model = Mock()
        
        # Создаем несколько mock boxes
        mock_box1 = Mock()
        mock_box1.cls = torch.tensor([0])
        mock_box1.conf = torch.tensor([0.85])
        mock_box1.xyxy = torch.tensor([[100, 100, 200, 200]])
        
        mock_box2 = Mock()
        mock_box2.cls = torch.tensor([1])
        mock_box2.conf = torch.tensor([0.92])
        mock_box2.xyxy = torch.tensor([[300, 300, 400, 400]])
        
        # Настраиваем мок результата
        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2]
        mock_result.names = {0: "person", 1: "car"}
        
        plot_result = np.zeros((300, 300, 3), dtype=np.uint8)
        mock_result.plot.return_value = plot_result
        
        mock_model.return_value = [mock_result]
        app.state.model = mock_model
        
        # Создаем тестовое изображение
        image = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        files = {"file": ("test.jpg", buffer, "image/jpeg")}
        response = client.post("/detect", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_objects"] == 2
        assert len(data["detections"]) == 2
