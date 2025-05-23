from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import base64
import io
from PIL import Image

from app.main import app

client = TestClient(app)

# Мокаем модель YOLO для всех тестов
@patch('app.main.YOLO')
def test_root_endpoint(mock_yolo):
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    assert "YOLO Object Detection API работает" in response.json()["message"]

@patch('app.main.YOLO')
def test_health_check(mock_yolo):
    """Тест проверки здоровья"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@patch('app.main.YOLO')
def test_detect_with_image(mock_yolo):
    """Тест детекции с изображением"""
    # Настраиваем мок
    mock_model = Mock()
    mock_box = Mock()
    mock_box.cls = [0]
    mock_box.conf = [0.85]
    mock_box.xyxy = [[100, 100, 200, 200]]
    
    mock_result = Mock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "person"}
    mock_result.plot.return_value = [[255, 0, 0]]  # Простой массив
    
    mock_model.return_value = [mock_result]
    mock_yolo.return_value = mock_model
    
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

@patch('app.main.YOLO')
def test_detect_base64(mock_yolo):
    """Тест детекции с base64"""
    # Настраиваем мок
    mock_model = Mock()
    mock_result = Mock()
    mock_result.boxes = []  # Нет объектов
    mock_result.plot.return_value = [[255, 0, 0]]
    
    mock_model.return_value = [mock_result]
    mock_yolo.return_value = mock_model
    
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
    text_content = b"This is not an image"
    files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
    
    response = client.post("/detect", files=files)
    assert response.status_code == 400
    assert "изображением" in response.json()["detail"]

def test_detect_base64_missing_image():
    """Тест без поля image"""
    payload = {"confidence": 0.5}
    response = client.post("/detect_base64", json=payload)
    assert response.status_code == 400
    assert "обязательно" in response.json()["detail"]
