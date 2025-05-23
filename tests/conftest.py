import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import torch
from PIL import Image
import io
import base64

# Импортируем приложение
from app.main import app

@pytest.fixture
def client():
    """Фикстура для тестового клиента FastAPI"""
    # Мокаем модель YOLO для тестов
    mock_model = Mock()
    
    # Настраиваем mock результаты
    mock_box = Mock()
    mock_box.cls = [0]  # класс объекта
    mock_box.conf = [0.85]  # уверенность
    mock_box.xyxy = [torch.tensor([100, 100, 200, 200])]  # координаты
    
    mock_result = Mock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person"}  # словарь классов
    mock_result.plot.return_value = torch.zeros((300, 300, 3), dtype=torch.uint8).numpy()
    
    mock_model.return_value = [mock_result]
    
    # Патчим состояние приложения
    with patch.object(app.state, 'model', mock_model):
        with TestClient(app) as test_client:
            yield test_client

@pytest.fixture
def sample_image_base64():
    """Фикстура для создания тестового изображения в base64"""
    # Создаем простое RGB изображение
    image = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@pytest.fixture
def sample_image_file():
    """Фикстура для создания тестового файла изображения"""
    image = Image.new('RGB', (100, 100), color='blue')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer
